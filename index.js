const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { OpenAI } = require('openai');
const axios = require('axios');
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
app.use(cors(ALLOWED_ORIGINS ? { origin: ALLOWED_ORIGINS.split(',').map((s) => s.trim()) } : {}));

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 8 * 1024 * 1024 } });


const CATEGORY_LABEL = {
  top: '👕 상의',
  bottom: '👖 하의',
  outer: '🧥 아우터',
  shoes: '👟 신발',
  bag: '👜 가방',
  acc: '🧢 액세서리',
};

/* ───────── 1. 이미지 → 옷 식별 (OpenAI Vision) ───────── */
async function detectItems(buffer, mime) {
  const dataUrl = `data:${mime};base64,${buffer.toString('base64')}`;
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      response_format: { type: 'json_object' },
      messages: [
        {
          role: 'system',
          content:
            '너는 패션 상품 식별기다. 사진 속 옷/신발/가방/액세서리를 각각 식별해 JSON으로만 답한다. ' +
            '형식: {"items":[{"category":"top|bottom|outer|shoes|bag|acc","type":"상품종류(예: 후드티, 와이드 데님팬츠)",' +
            '"color":"색상","brand":"로고/텍스트가 또렷이 읽힐 때만, 아니면 null","keywords":["핏/소재/스타일 키워드 최대 4개"],"confidence":0~1}]}. ' +
            '브랜드는 절대 추측하지 마라. 옷이 없으면 items를 빈 배열로.',
        },
        { role: 'user', content: [{ type: 'image_url', image_url: { url: dataUrl } }] },
      ],
    }),
  });
  if (!res.ok) throw new Error(`OpenAI ${res.status}`);
  const json = await res.json();
  const parsed = JSON.parse(json.choices?.[0]?.message?.content ?? '{}');
  return (Array.isArray(parsed.items) ? parsed.items : [])
    .filter((i) => CATEGORY_LABEL[i.category] && i.type)
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
}

/* ───────── 2. 네이버 쇼핑 검색 (상품 상세 링크 포함) ───────── */
const stripTags = (s = '') => s.replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&quot;/g, '"');

async function searchNaver(query) {
  const url = new URL('https://openapi.naver.com/v1/search/shop.json');
  url.searchParams.set('query', query);
  url.searchParams.set('display', '30');
  url.searchParams.set('sort', 'sim');
  const res = await fetch(url, {
    headers: { 'X-Naver-Client-Id': NAVER_CLIENT_ID, 'X-Naver-Client-Secret': NAVER_CLIENT_SECRET },
  });
  if (!res.ok) throw new Error(`Naver ${res.status}`);
  return (await res.json()).items ?? [];
}

/**
 * 제휴(CPS) 링크 변환 지점.
 * 네이버 API의 item.link 는 해당 상품의 상세 페이지 링크다(쇼핑몰 메인이 아님).
 * 실제 수수료 정산용 링크로 바꾸려면 여기서 제휴 네트워크(쿠팡파트너스 등)의 변환 API를 호출하세요.
 */
const toAffiliateUrl = (link) => link;

// 키워드 겹침 비율 기반 "추정" 유사도 (이미지 임베딩 비교가 아님)
function estimateSimilarity(title, tokens) {
  if (!tokens.length) return 0;
  const t = title.replace(/\s/g, '').toLowerCase();
  const hit = tokens.filter((k) => t.includes(k.replace(/\s/g, '').toLowerCase())).length;
  return Math.min(99, Math.round((hit / tokens.length) * 100));
}

async function recommend(item, mode) {
  const tokens = [item.color, item.type, ...(item.keywords ?? [])].filter(Boolean);
  const query =
    mode === 'exact' && item.brand
      ? [item.brand, item.color, item.type].filter(Boolean).join(' ')
      : tokens.slice(0, 4).join(' ');

  const raw = await searchNaver(query);
  const seen = new Set();
  const out = [];
  for (const it of raw) {
    const price = Number(it.lprice);
    if (!price || seen.has(it.productId)) continue;
    seen.add(it.productId);
    const name = stripTags(it.title);
    if (mode === 'exact' && item.brand && !name.toLowerCase().includes(item.brand.toLowerCase())) continue;
    out.push({
      id: `naver-${it.productId}`,
      categoryId: item.category,
      platform: it.mallName || '네이버쇼핑',
      brand: it.brand || it.maker || '',
      name,
      price,
      shippingFee: null, // 네이버 쇼핑 API는 배송비를 제공하지 않음
      originalPrice: Number(it.hprice) > price ? Number(it.hprice) : undefined,
      imageUrl: it.image,
      purchaseUrl: toAffiliateUrl(it.link),
      similarity: mode === 'dupe' ? estimateSimilarity(name, tokens) : undefined,
    });
    if (out.length >= 12) break;
  }
  return out;
}

/* ───────── 3. API ───────── */
// search_id 별 식별 결과 캐시(10분). 탭/카테고리 전환 때 비전 API를 다시 호출하지 않음.
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
      if (!req.file) return res.status(400).json({ error: 'image required' });
      detected = await detectItems(req.file.buffer, req.file.mimetype);
      if (searchId) cache.set(searchId, { items: detected, at: Date.now() });
    }
    if (!detected.length) return res.json({ categories: [], recommendations: [], sponsored: null });

    const target = detected.find((i) => i.category === req.body.category) ?? detected[0];
    const recommendations = await recommend(target, mode);

    res.json({
      categories: detected.map((i) => ({
        id: i.category,
        label: `${CATEGORY_LABEL[i.category]} (${Math.round((i.confidence ?? 0) * 100)}%)`,
        confidence: i.confidence ?? 0,
      })),
      recommendations,
      sponsored: null, // 광고 연동 시 SponsoredAd 객체를 내려주면 화면에 노출됨
    });
  } catch (e) {
    console.error('[analyze]', e);
    res.status(502).json({ error: 'analyze failed' });
  }
});

// 가격 하락 알림: 현재는 메모리 저장 스텁. 실제 푸시 발송은 DB + 토스 푸시 서버 API 연동이 필요.
const alerts = new Map();
app.post('/api/v1/chakchak/alerts', (req, res) => {
  alerts.set(req.body.itemId, req.body.currentFinalPrice);
  res.json({ ok: true });
});
app.delete('/api/v1/chakchak/alerts', (req, res) => {
  alerts.delete(req.body.itemId);
  res.json({ ok: true });
});

app.listen(PORT, () => console.log(`chakchak server on :${PORT}`));
