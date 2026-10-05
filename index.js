const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const multer = require('multer');

dotenv.config();

const {
  PORT = 3000,
  OPENAI_API_KEY,
  NAVER_CLIENT_ID,
  NAVER_CLIENT_SECRET,
  ALLOWED_ORIGINS = '',
} = process.env;

const app = express();

app.use(express.json({ limit: '10mb' }));
app.use(
  cors(
    ALLOWED_ORIGINS
      ? { origin: ALLOWED_ORIGINS.split(',').map((s) => s.trim()) }
      : { origin: '*' }
  )
);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 },
});

const CATEGORY_LABEL = {
  top: '👕 상의',
  bottom: '👖 하의',
  outer: '🧥 아우터',
  shoes: '👟 신발',
  bag: '👜 가방',
  acc: '🧢 액세서리',
};

/* ───────── 1. OpenAI GPT-4o Vision API (패션 고도화 분석) ───────── */
async function detectItems(buffer, mime) {
  const dataUrl = `data:${mime};base64,${buffer.toString('base64')}`;

  const systemPrompt = `
너는 대한민국 최고의 패션 커머스 AI 분석기다.
사용자가 업로드한 이미지 속 착장(옷/신발/가방/액세서리)을 매우 정밀하게 시각 분석하여, 네이버 쇼핑 및 패션 플랫폼(무신사, 지그재그, 에이블리, 크림 등)에서 **실제 동일 상품 또는 최적의 가성비 대체재(Dupe)**를 찾을 수 있는 정교한 한국어 검색어를 생성한다.

[반환 JSON 구조]
{
  "items": [
    {
      "category": "top|bottom|outer|shoes|bag|acc",
      "type": "상세 아이템 종류 (예: 와이드 데님 팬츠, 오버핏 헤비 후드티, 크롭 레더 자켓)",
      "color": "정확한 대표 색상 (예: 흑청, 차콜, 아이보리, 빈티지 브라운)",
      "brand": "이미지에서 확증 가능한 로고/텍스트가 있을 때만 브랜드명, 없으면 null (절대 추측 금지)",
      "fit_and_details": ["핏/소재/디테일 특징 (예: 카고 포켓, 투웨이 지퍼, 골지, 크롭, 오버핏)"],
      "exact_search_query": "동일/동급 원본 제품을 찾기 위한 정밀 검색어 (브랜드가 있다면 브랜드 포함, 아니면 세부 디테일 포함)",
      "dupe_search_query": "비슷한 디자인의 가성비 싼 대체재 상품을 찾기 위한 검색어 (브랜드 제외, 색상+핏+핵심디테일+종류 위주)",
      "confidence": 0.0 ~ 1.0
    }
  ]
}

[필수 규칙]
1. 절대로 존재하지 않는 브랜드를 지어내거나 추측하지 마라.
2. 이미지 속 모든 패션 아이템을 감지하되, 카테고리는 [top, bottom, outer, shoes, bag, acc] 중 하나로 지정하라.
3. 옷이 식별되지 않으면 items를 빈 배열 []로 반환하라.
`;

  try {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o', // 시각적 정밀도가 높은 gpt-4o 적용
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: systemPrompt },
          {
            role: 'user',
            content: [{ type: 'image_url', image_url: { url: dataUrl } }],
          },
        ],
        temperature: 0.1,
      }),
    });

    if (!res.ok) {
      const errText = await res.text();
      console.error('[OpenAI Error]', res.status, errText);
      throw new Error(`OpenAI API Error: ${res.status}`);
    }

    const json = await res.json();
    const parsed = JSON.parse(json.choices?.[0]?.message?.content ?? '{}');
    const items = Array.isArray(parsed.items) ? parsed.items : [];

    return items
      .filter((i) => CATEGORY_LABEL[i.category] && i.type)
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  } catch (error) {
    console.error('[detectItems Exception]', error);
    throw error;
  }
}

/* ───────── 2. 네이버 쇼핑 검색 API & 상품 구매 직링크 ───────── */
const stripTags = (s = '') =>
  s.replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&quot;/g, '"');

async function searchNaver(query, sortMode = 'sim') {
  if (!NAVER_CLIENT_ID || !NAVER_CLIENT_SECRET || NAVER_CLIENT_ID === 'none') {
    console.warn('[Naver API Warning] Naver Client ID/Secret 이 설정되지 않았거나 "none" 상태입니다.');
    return [];
  }

  try {
    const url = new URL('https://openapi.naver.com/v1/search/shop.json');
    url.searchParams.set('query', query);
    url.searchParams.set('display', '40');
    url.searchParams.set('sort', sortMode); // sim: 정확도순, asc: 낮은 가격순

    const res = await fetch(url, {
      headers: {
        'X-Naver-Client-Id': NAVER_CLIENT_ID,
        'X-Naver-Client-Secret': NAVER_CLIENT_SECRET,
      },
    });

    if (!res.ok) {
      console.error('[Naver API Error]', res.status, await res.text());
      return [];
    }

    const data = await res.json();
    return data.items ?? [];
  } catch (err) {
    console.error('[searchNaver Exception]', err);
    return [];
  }
}

function estimateSimilarity(title, keywords) {
  if (!keywords || !keywords.length) return 80;
  const t = title.replace(/\s/g, '').toLowerCase();
  let hits = 0;
  keywords.forEach((k) => {
    if (k && t.includes(k.replace(/\s/g, '').toLowerCase())) hits++;
  });
  const score = Math.round((hits / keywords.length) * 35) + 65;
  return Math.min(99, Math.max(65, score));
}

async function recommend(item, mode) {
  let searchQuery = '';
  let sortMode = 'sim';

  if (mode === 'exact') {
    // 원본 옷 검색: 브랜드 + 색상 + 세부 아이템
    searchQuery =
      item.exact_search_query ||
      [item.brand, item.color, item.type].filter(Boolean).join(' ');
    sortMode = 'sim';
  } else {
    // 가성비/비슷한 싼 옷 검색 (Dupe): 브랜드 제외, 디자인 디테일 + 핏 + 색상 위주
    searchQuery =
      item.dupe_search_query ||
      [item.color, item.type, ...(item.fit_and_details || [])].filter(Boolean).slice(0, 3).join(' ');
    sortMode = 'asc'; // 낮은 가격순
  }

  console.log(`[Search Query] Mode: ${mode} | Query: "${searchQuery}"`);

  let raw = await searchNaver(searchQuery, sortMode);

  // 검색 결과가 적으면 검색어를 단순화하여 재검색 (Fallback)
  if (raw.length < 5) {
    const fallbackQuery = [item.color, item.type].filter(Boolean).join(' ');
    console.log(`[Fallback Query] "${fallbackQuery}"`);
    const fallbackRaw = await searchNaver(fallbackQuery, 'sim');
    raw = [...raw, ...fallbackRaw];
  }

  const seen = new Set();
  const out = [];
  const keywordTokens = [item.color, item.type, ...(item.fit_and_details || [])].filter(Boolean);

  for (const it of raw) {
    const price = Number(it.lprice);
    if (!price || seen.has(it.productId)) continue;
    seen.add(it.productId);

    const name = stripTags(it.title);

    // it.link 는 해당 상품의 상세 구매 페이지(스마트스토어, 무신사, 에이블리 등)로 연결되는 직링크입니다.
    const productDetailUrl = it.link;

    out.push({
      id: `naver-${it.productId}`,
      categoryId: item.category,
      platform: it.mallName || '네이버쇼핑',
      brand: it.brand || it.maker || '',
      name: name,
      price: price,
      shippingFee: null,
      originalPrice: Number(it.hprice) > price ? Number(it.hprice) : undefined,
      imageUrl: it.image,
      purchaseUrl: productDetailUrl, // 상세 상품 직링크
      similarity: estimateSimilarity(name, keywordTokens),
    });

    if (out.length >= 15) break;
  }

  // dupe 모드일 경우 가격이 저렴한 순으로 최종 정렬
  if (mode === 'dupe') {
    out.sort((a, b) => a.price - b.price);
  }

  return out;
}

/* ───────── 3. API 엔드포인트 ───────── */
const cache = new Map();
const TTL = 10 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of cache) if (now - v.at > TTL) cache.delete(k);
}, 60 * 1000).unref();

app.post('/api/v1/chakchak/analyze', upload.single('fashion_image'), async (req, res) => {
  try {
    const mode = req.body.mode === 'dupe' ? 'dupe' : 'exact';
    const searchId = String(req.body.search_id || '');
    let detected = cache.get(searchId)?.items;

    if (!detected) {
      if (!req.file) {
        return res.status(400).json({ error: 'fashion_image 파일이 필요합니다.' });
      }
      detected = await detectItems(req.file.buffer, req.file.mimetype);
      if (searchId && detected.length > 0) {
        cache.set(searchId, { items: detected, at: Date.now() });
      }
    }

    if (!detected.length) {
      return res.json({ categories: [], recommendations: [], sponsored: null });
    }

    const requestedCategory = req.body.category;
    const target = detected.find((i) => i.category === requestedCategory) ?? detected[0];
    const recommendations = await recommend(target, mode);

    res.json({
      categories: detected.map((i) => ({
        id: i.category,
        label: `${CATEGORY_LABEL[i.category]} (${Math.round((i.confidence ?? 0) * 100)}%)`,
        confidence: i.confidence ?? 0,
      })),
      recommendations,
      sponsored: null,
    });
  } catch (e) {
    console.error('[API Error /analyze]', e);
    res.status(500).json({ error: e.message || 'analyze failed' });
  }
});

app.post('/api/v1/chakchak/alerts', (req, res) => res.json({ ok: true }));
app.delete('/api/v1/chakchak/alerts', (req, res) => res.json({ ok: true }));

app.listen(PORT, () => console.log(`chakchak server running on port ${PORT}`));
