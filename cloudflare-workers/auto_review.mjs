// Conservative automated quality gate for the isolated Quantum knowledge crawler.
// Deterministic screening only: this is not an LLM judgment or factual verification.
const ADULT = /포르노|성인.?영상|성행위|성적.?착취|아동.?성착취/i;
const LOW_VALUE = /고등학교|중학교|초등학교|기업|주식회사|그룹 AG|영화|소설|음반|노래|선거구|역대|목록|분류|연도별|동음이의어/i;
const SENSITIVE = /주민등록번호|여권번호|전화번호|휴대전화|이메일 주소|집 주소|개인정보/i;
const TOPIC = /(학|과학|수학|물리|화학|생물|천문|지질|경제|정치|사회|언어|철학|역사|기술|공학|컴퓨터|정보|알고리즘|자료구조|환경|기후|교육|법학|논리|문법|통계|확률|금융|건축|에너지|생태|의학|해부|생리|심리|문학|지리|우주|원자|분자|유전|진화|민주주의|국제 관계)/;
export function reviewCandidate(row) {
  const title = String(row.title || "").trim();
  const text = String(row.content || "").trim();
  const url = String(row.url || "");
  const domain = String(row.domain || "");
  let score = 0;
  const reasons = [];
  if (ADULT.test(title + " " + text)) return { status: "excluded", score: 0, reason: "성인 콘텐츠 주제 제외" };
  if (SENSITIVE.test(text)) return { status: "pending_review", score: 0, reason: "개인정보 가능성으로 사람 검토 필요" };
  if (domain !== "ko.wikipedia.org" || !url.startsWith("https://ko.wikipedia.org/wiki/"))
    return { status: "pending_review", score: 0, reason: "허용된 출처 도메인 확인 실패" };
  if (row.license_status !== "attribution_required" || row.license_name !== "CC BY-SA 4.0" ||
      !String(row.attribution || "").trim() || !row.revision_id)
    return { status: "pending_review", score: 0, reason: "라이선스·출처·리비전 메타데이터 불충분" };
  score += 30; reasons.push("출처·라이선스 메타데이터");
  if (text.length >= 1800 && text.length <= 50000) { score += 20; reasons.push("본문 길이 적정"); }
  else return { status: "pending_review", score, reason: "본문 길이 또는 완결성 검토 필요" };
  if (LOW_VALUE.test(title)) return { status: "pending_review", score, reason: "개별 인물·기관·작품·목록형 항목은 우선 대기" };
  if (TOPIC.test(title)) { score += 20; reasons.push("일반 지식 주제"); }
  else return { status: "pending_review", score, reason: "범용 학습 주제인지 분류 불확실" };
  const first = text.slice(0, 500);
  if (/(은|는|이란|란|분야이다|학문이다|방법이다|체계이다|개념이다|연구한다|가리킨다|말한다)/.test(first)) {
    score += 15; reasons.push("정의형 도입부");
  } else return { status: "pending_review", score, reason: "정의형 도입부 확인 필요" };
  const paragraphs = text.split(/\n{2,}/).filter(x => x.trim().length > 50).length;
  if (paragraphs >= 3) { score += 10; reasons.push("문단 구성"); }
  else return { status: "pending_review", score, reason: "본문 구조가 짧거나 불완전함" };
  const segments = text.split(/[.!?。\n]/).filter(x => x.trim().length > 20);
  const unique = new Set(segments.map(x => x.trim()));
  const duplicateRatio = segments.length ? 1 - unique.size / segments.length : 1;
  if (duplicateRatio > 0.25) return { status: "pending_review", score, reason: "중복 문장 비율이 높음" };
  score += 5;
  if (score >= 90) return { status: "approved", score, reason: "자동 품질 기준 통과: " + reasons.join(", ") };
  return { status: "pending_review", score, reason: "자동 승인 점수 기준 미달: " + reasons.join(", ") };
}
