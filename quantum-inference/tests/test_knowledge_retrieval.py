import sys
import sqlite3
import unittest
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knowledge_retrieval import build_fts_query, query_terms, rank_documents, score_document, search_knowledge_index


class KnowledgeRetrievalTests(unittest.TestCase):
    def test_exact_title_is_top_score(self):
        self.assertEqual(score_document("대한민국", "대한민국", "한반도에 위치한 국가"), 1.0)

    def test_korean_particles_are_normalized(self):
        self.assertEqual(query_terms("태양계의 행성으로"), ["태양계", "행성"])
        score = score_document("인공지능이란 무엇인가요?", "인공지능", "컴퓨터 과학의 한 분야")
        self.assertGreaterEqual(score, 0.9)

    def test_common_korean_nouns_ending_in_particle_strings_are_preserved(self):
        self.assertEqual(query_terms("국가"), ["국가"])
        self.assertEqual(query_terms("사과"), ["사과"])
        self.assertEqual(query_terms("회의"), ["회의"])
        self.assertEqual(query_terms("전문가"), ["전문가"])

    def test_outer_particles_are_removed_without_corrupting_protected_nouns(self):
        self.assertEqual(query_terms("국가의 역사"), ["국가", "역사"])
        self.assertEqual(query_terms("사과를 먹는 방법"), ["사과", "먹"])
        self.assertEqual(query_terms("회의에서 결정한 결과"), ["회의", "결정한", "결과"])

    def test_protected_korean_nouns_survive_attached_outer_particles(self):
        examples = {
            "아이의": "아이",
            "국가의": "국가",
            "사과를": "사과",
            "결과는": "결과",
            "효과가": "효과",
            "교과의": "교과",
            "허가를": "허가",
            "추가로": "추가",
            "참가가": "참가",
            "평가를": "평가",
            "전문가의": "전문가",
            "작가가": "작가",
            "화가의": "화가",
            "원가의": "원가",
            "물가가": "물가",
            "주가의": "주가",
            "대가를": "대가",
            "상가의": "상가",
            "휴가를": "휴가",
            "증가가": "증가",
            "감소가": "감소",
            "국어를": "국어",
            "영어를": "영어",
            "언어의": "언어",
            "회의에서": "회의",
            "국가에서는": "국가",
            "회의에서의": "회의",
            "전문가에게는": "전문가",
        }
        for attached, expected in examples.items():
            with self.subTest(attached=attached):
                self.assertEqual(query_terms(attached), [expected])

    def test_definition_remains_a_searchable_lexical_noun(self):
        self.assertEqual(query_terms("정의"), ["정의"])
        self.assertEqual(query_terms("정의의"), ["정의"])

    def test_common_lexical_nouns_are_not_stripped_as_particles(self):
        examples = {
            "종이는": "종이",
            "사이에서": "사이",
            "차이가": "차이",
            "고양이는": "고양이",
            "길이는": "길이",
            "높이가": "높이",
            "깊이는": "깊이",
            "먹이를": "먹이",
            "놀이를": "놀이",
            "추이는": "추이",
            "정의의": "정의",
            "강의에서": "강의",
            "합의가": "합의",
            "건의는": "건의",
            "질의에": "질의",
            "논의에서": "논의",
            "동의가": "동의",
            "주의를": "주의",
            "편의로": "편의",
            "임의로": "임의",
            "협의는": "협의",
            "항의가": "항의",
            "심의에서": "심의",
            "결의가": "결의",
            "상의는": "상의",
            "성과가": "성과",
            "학과에서": "학과",
            "전과는": "전과",
            "도로로": "도로",
            "가로로": "가로",
            "세로로": "세로",
            "난로에": "난로",
            "경로를": "경로",
            "진로는": "진로",
            "회로의": "회로",
            "통로로": "통로",
            "미로에서": "미로",
            "피로가": "피로",
            "마을에서": "마을",
            "가을에": "가을",
            "바늘이": "바늘",
            "마늘을": "마늘",
        }
        for attached, expected in examples.items():
            with self.subTest(attached=attached):
                self.assertEqual(query_terms(attached), [expected])

    def test_additional_korean_postpositions_are_removed_conservatively(self):
        examples = {
            "국가로부터": "국가",
            "회의에서부터": "회의",
            "전문가에게서": "전문가",
            "국가까지": "국가",
            "사과처럼": "사과",
            "결과보다": "결과",
            "회의에서부터의": "회의",
        }
        for attached, expected in examples.items():
            with self.subTest(attached=attached):
                self.assertEqual(query_terms(attached), [expected])

    def test_fts_query_uses_prefix_matching(self):
        self.assertEqual(build_fts_query("태양계의 행성으로"), '"태양계"* OR "행성"*')

    def test_additional_postpositions_reach_matching_fts5_documents(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, text, tokenize='unicode61')")
            conn.executemany(
                "INSERT INTO docs(title,text) VALUES(?,?)",
                [
                    ("국가", "국가의 역사와 제도"),
                    ("회의", "회의의 진행과 기록"),
                    ("전문가", "전문가의 연구와 의견"),
                ],
            )
            for question, expected_title in (
                ("국가로부터", "국가"),
                ("회의에서부터", "회의"),
                ("전문가에게서", "전문가"),
            ):
                with self.subTest(question=question):
                    rows = conn.execute(
                        "SELECT title FROM docs WHERE docs MATCH ?",
                        (build_fts_query(question),),
                    ).fetchall()
                    self.assertIn((expected_title,), rows)
        finally:
            conn.close()

    def test_fts5_finds_protected_nouns_after_stacked_particles(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, text, tokenize='unicode61')")
            conn.execute(
                "INSERT INTO docs(title,text) VALUES(?,?)",
                ("국가", "국가의 역사와 제도"),
            )
            rows = conn.execute(
                "SELECT title FROM docs WHERE docs MATCH ?",
                (build_fts_query("국가에서는 어떤 제도가 있나요"),),
            ).fetchall()
            self.assertEqual(rows, [("국가",)])
        finally:
            conn.close()

    def test_single_syllable_nouns_survive_particle_removal(self):
        terms = query_terms("별과 달은 무엇인가요?")
        self.assertIn("별", terms)
        self.assertIn("달", terms)
        self.assertIn('"별"*', build_fts_query("별과 달은 무엇인가요?"))

    def test_conservative_domain_alias_expansion(self):
        self.assertIn("화학", query_terms("화학자에게 수여하는 상"))
        self.assertIn("아날로그", query_terms("연속 신호를 다루는 전자 시스템"))
        self.assertIn("미적분학", query_terms("함수의 변화율과 누적량"))
        self.assertIn("입자물리학", query_terms("기본 입자와 물리학"))
 
    def test_short_generic_title_does_not_receive_phrase_bonus(self):
        score = score_document(
            "물질을 이루는 기본 입자와 상호작용을 연구하는 물리학",
            "물리학",
            "물리학은 자연 현상을 연구하는 학문이다",
        )
        self.assertLess(score, 0.5)

    def test_fts5_finds_korean_words_with_attached_endings(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, text, tokenize='unicode61')")
            conn.execute(
                "INSERT INTO docs(title,text) VALUES(?,?)",
                ("태양계", "태양계와 행성으로 구성된 천체계"),
            )
            rows = conn.execute(
                "SELECT title,text FROM docs WHERE docs MATCH ?",
                (build_fts_query("태양계의 행성"),),
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "태양계")
        finally:
            conn.close()

    def test_unrelated_or_match_is_filtered(self):
        rows = [{
            "title": "경제학",
            "url": "https://ko.wikipedia.org/wiki/경제학",
            "text": "경제 활동과 자원의 배분을 연구하는 학문",
        }]
        self.assertEqual(rank_documents("태양계의 행성", rows), [])

    def test_single_accidental_body_word_is_filtered(self):
        rows = [{
            "title": "경제학", "url": "a",
            "text": "태양계라는 단어가 우연히 포함된 설명",
        }]
        self.assertEqual(rank_documents("태양계", rows), [])

    def test_relevant_title_ranks_above_body_only_match(self):
        rows = [
            {"title": "경제학", "url": "a", "text": "태양계라는 단어가 우연히 포함된 설명"},
            {"title": "태양계", "url": "b", "text": "태양은 태양계 중심의 항성이다"},
        ]
        ranked = rank_documents("태양계란 무엇인가", rows)
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["title"], "태양계")

    def test_multi_term_body_evidence_can_be_kept(self):
        rows = [{
            "title": "천문학 개론", "url": "c",
            "text": "태양계 행성의 공전과 태양의 중력에 관한 설명",
        }]
        self.assertEqual(len(rank_documents("태양계 행성의 공전", rows)), 1)

    def test_ranked_results_explain_title_and_body_match_evidence(self):
        rows = [{
            "title": "태양계",
            "url": "https://ko.wikipedia.org/wiki/태양계",
            "text": "태양계 행성의 공전과 태양의 중력",
        }]
        result = rank_documents("태양계 행성의 공전", rows)[0]
        trace = result["match_diagnostics"]
        self.assertEqual(trace["query_terms"], ["태양계", "행성", "공전"])
        self.assertEqual(trace["title_matched_terms"], ["태양계"])
        self.assertEqual(trace["body_matched_terms"], ["태양계", "행성", "공전"])
        self.assertEqual(trace["title_match_count"], 1)
        self.assertEqual(trace["body_match_count"], 3)
        self.assertEqual(trace["title_coverage"], 0.333333)
        self.assertEqual(trace["body_coverage"], 1.0)
        self.assertEqual(trace["evidence_basis"], "title")

    def test_body_only_match_diagnostic_requires_multiple_terms(self):
        rows = [{
            "title": "천문학 개론",
            "url": "https://ko.wikipedia.org/wiki/천문학",
            "text": "태양계 행성의 공전과 태양의 중력",
        }]
        result = rank_documents("태양계 행성의 공전", rows)[0]
        trace = result["match_diagnostics"]
        self.assertEqual(trace["title_matched_terms"], [])
        self.assertEqual(trace["body_matched_terms"], ["태양계", "행성", "공전"])
        self.assertEqual(trace["evidence_basis"], "multiple_body_terms")

    def test_ten_subject_areas_have_relevant_title_matches(self):
        cases = [
            ("조선 왕조의 역사", "조선", "한국의 역사 속 왕조"),
            ("광합성 과정 설명", "광합성", "식물이 빛 에너지를 이용하는 과정"),
            ("일차방정식 풀이", "일차방정식", "수학의 방정식"),
            ("시장경제 원리", "시장경제", "경제 체제의 한 형태"),
            ("민주주의 제도", "민주주의", "정치 체제"),
            ("아시아의 지리", "아시아", "세계의 대륙"),
            ("컴퓨터 프로그래밍", "프로그래밍", "컴퓨터 프로그램 작성"),
            ("한국어 문법", "한국어", "한국에서 사용하는 언어"),
            ("태양계 행성", "태양계", "태양과 행성으로 구성된 천체계"),
            ("인공지능의 원리", "인공지능", "컴퓨터 과학의 한 분야"),
        ]
        for question, title, body in cases:
            with self.subTest(question=question):
                ranked = rank_documents(question, [{"title": title, "text": body, "url": title}])
                self.assertTrue(ranked)
                self.assertEqual(ranked[0]["title"], title)

    def test_duplicate_documents_are_removed(self):
        row = {"title": "태양계", "url": "same", "text": "태양계 행성"}
        self.assertEqual(len(rank_documents("태양계 행성", [row, row])), 1)

    def test_result_limit_is_respected(self):
        rows = [
            {"title": f"태양계 {i}", "url": str(i), "text": "태양계 행성"}
            for i in range(5)
        ]
        self.assertEqual(len(rank_documents("태양계 행성", rows, limit=2)), 2)

    def test_query_rejects_non_string_input(self):
        for value in (None, 123, True, ["태양계"]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    query_terms(value)
                with self.assertRaises(TypeError):
                    build_fts_query(value)

    def test_query_length_is_bounded_before_tokenization(self):
        oversized = "태양계 " * 1001
        with self.assertRaisesRegex(ValueError, "2000자를 초과"):
            query_terms(oversized)
        with self.assertRaisesRegex(ValueError, "2000자를 초과"):
            build_fts_query(oversized)

    def test_query_terms_are_bounded_to_sixty_four_unique_terms(self):
        question = " ".join(f"term{index}" for index in range(100))
        terms = query_terms(question)
        self.assertEqual(len(terms), 64)
        self.assertEqual(terms[0], "term0")
        self.assertEqual(terms[-1], "term63")
        self.assertEqual(len(set(terms)), 64)

    def test_query_term_bound_keeps_first_occurrence_order(self):
        question = "태양계 행성 태양계 달 " + " ".join(
            f"topic{index}" for index in range(70)
        )
        terms = query_terms(question)
        self.assertEqual(terms[:4], ["태양계", "행성", "달", "topic0"])
        self.assertEqual(len(terms), 64)

    def test_fts_term_limit_requires_bounded_integer(self):
        for value in (True, 1.5, "5", 0, 21):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_fts_query("태양계", max_terms=value)

    def test_rank_limit_requires_bounded_integer(self):
        rows = [{"title": "태양계", "text": "태양계 행성", "url": "x"}]
        for value in (True, 1.5, "2", 0, 101):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    rank_documents("태양계", rows, limit=value)

    def test_min_score_rejects_non_finite_and_out_of_range_values(self):
        rows = [{"title": "태양계", "text": "태양계 행성", "url": "x"}]
        for value in (True, "0.2", float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    rank_documents("태양계", rows, min_score=value)

    def test_search_document_fields_must_be_strings(self):
        for row in (
            {"title": 123, "text": "태양계 행성", "url": "x"},
            {"title": "태양계", "text": None, "url": "x"},
            {"title": "태양계", "text": "태양계 행성", "url": 42},
        ):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "문자열"):
                    rank_documents("태양계", [row])

    def test_rows_must_be_an_explicit_sequence(self):
        for rows in ("invalid", iter([]), None):
            with self.subTest(rows_type=type(rows).__name__):
                with self.assertRaises(ValueError):
                    rank_documents("태양계", rows)

    def test_stopword_only_question_returns_no_results(self):
        self.assertEqual(rank_documents("무엇인가요 어떻게 설명", []), [])


    def _knowledge_connection(self, include_fts=True):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE knowledge_docs(page_id TEXT, title, url, text, attribution TEXT, license TEXT, revision_id TEXT, content_hash TEXT)")
        conn.execute(
            "INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?,?,?)",
            ("1", "태양계", "https://ko.wikipedia.org/wiki/태양계",
             "태양계 행성은 태양 주위를 공전한다", "태양계 - https://ko.wikipedia.org/wiki/태양계 CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/", "CC BY-SA 4.0", "12345", hashlib.sha256("태양계 행성은 태양 주위를 공전한다".encode("utf-8")).hexdigest()),
        )
        if include_fts:
            conn.execute("CREATE VIRTUAL TABLE knowledge_fts USING fts5(page_id UNINDEXED,title,text)")
            conn.execute("INSERT INTO knowledge_fts VALUES(?,?,?)",
                         ("1", "태양계", "태양계 행성은 태양 주위를 공전한다"))
        return conn

    def test_index_search_uses_fts_and_returns_diagnostic_state(self):
        conn = self._knowledge_connection()
        try:
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["used_fallback"])
            self.assertEqual(result["documents"][0]["title"], "태양계")
            self.assertEqual(result["documents"][0]["page_id"], "1")
            self.assertEqual(result["documents"][0]["revision_id"], "12345")
        finally:
            conn.close()

    def test_index_search_falls_back_when_fts_table_is_missing(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "fts_fallback")
            self.assertTrue(result["used_fallback"])
            self.assertEqual(result["documents"][0]["title"], "태양계")
        finally:
            conn.close()

    def test_invalid_high_ranked_rows_do_not_hide_valid_rows_below_result_limit(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            for index in range(4):
                title = "태양계 행성"
                url = f"https://ko.wikipedia.org/wiki/태양계_행성_{index}"
                body = "태양계 행성에 관한 문서 내용입니다."
                conn.execute(
                    "INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?,?,?)",
                    (
                        str(index + 10), title, url, body,
                        "불완전한 출처 표시", "CC BY-SA 4.0", str(20000 + index),
                        hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    ),
                )
            result = search_knowledge_index(conn, "태양계 행성", limit=1)
            self.assertEqual(result["status"], "fts_fallback")
            self.assertEqual(len(result["documents"]), 1)
            self.assertEqual(result["documents"][0]["page_id"], "1")
            self.assertEqual(result["rejected_source_rows"], 4)
        finally:
            conn.close()

    def test_fts_candidate_window_keeps_valid_rows_beyond_old_fifty_row_cap(self):
        conn = self._knowledge_connection(include_fts=True)
        try:
            for index in range(60):
                title = "태양계 행성"
                body = "태양계 행성에 관한 검증되지 않은 문서입니다."
                url = f"https://ko.wikipedia.org/wiki/잘못된_문서_{index}"
                conn.execute(
                    "INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?,?,?)",
                    (
                        str(index + 100), title, url, body,
                        "불완전한 출처 표시", "CC BY-SA 4.0", str(30000 + index),
                        hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    ),
                )
                conn.execute(
                    "INSERT INTO knowledge_fts VALUES(?,?,?)",
                    (str(index + 100), title, body),
                )
            result = search_knowledge_index(conn, "태양계 행성", limit=1)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(result["documents"]), 1)
            self.assertEqual(result["documents"][0]["page_id"], "1")
            self.assertEqual(result["rejected_source_rows"], 60)
        finally:
            conn.close()

    def test_search_limit_requires_bounded_integer(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            for value in (True, 0, 11, 1.5, "3"):
                with self.subTest(limit=value):
                    with self.assertRaises(ValueError):
                        search_knowledge_index(conn, "태양계", limit=value)
        finally:
            conn.close()

    def test_index_search_rejects_rows_without_valid_source_attribution(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            conn.execute("UPDATE knowledge_docs SET attribution='저작자 표시만 있음'")
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "source_metadata_invalid")
            self.assertEqual(result["documents"], [])
            self.assertEqual(result["rejected_source_rows"], 1)
        finally:
            conn.close()

    def test_index_search_keeps_valid_attribution_and_reports_rejected_rows(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            conn.execute(
                "INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?,?,?)",
                ("2", "태양계", "https://example.com/not-wikipedia",
                 "태양계 행성 설명", "저작자 표시", "CC BY-SA 4.0", "12346", hashlib.sha256("태양계 행성 설명".encode("utf-8")).hexdigest()),
            )
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "fts_fallback")
            self.assertEqual(len(result["documents"]), 1)
            self.assertEqual(result["rejected_source_rows"], 1)
            self.assertEqual(result["documents"][0]["url"],
                             "https://ko.wikipedia.org/wiki/태양계")
        finally:
            conn.close()

    def test_index_search_reports_invalid_rows_without_coercion(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            conn.execute("UPDATE knowledge_docs SET title=123")
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "index_invalid")
            self.assertEqual(result["documents"], [])
        finally:
            conn.close()

    def test_index_search_rejects_tampered_content_hash(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            conn.execute("UPDATE knowledge_docs SET content_hash=?", ("0" * 64,))
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "document_integrity_invalid")
            self.assertEqual(result["documents"], [])
            self.assertEqual(result["rejected_integrity_rows"], 1)
        finally:
            conn.close()

    def test_index_search_rejects_missing_revision_metadata(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            conn.execute("UPDATE knowledge_docs SET revision_id=NULL")
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "document_integrity_invalid")
            self.assertEqual(result["documents"], [])
            self.assertEqual(result["rejected_integrity_rows"], 1)
        finally:
            conn.close()

    def test_index_search_distinguishes_mixed_source_and_integrity_rejections(self):
        conn = self._knowledge_connection(include_fts=False)
        try:
            conn.execute(
                "UPDATE knowledge_docs SET attribution='잘못된 출처 정보'"
            )
            title = "태양계 행성"
            url = "https://ko.wikipedia.org/wiki/태양계_행성"
            body = "태양계 행성에 관한 추가 문서 내용입니다."
            attribution = (
                f"Wikipedia contributors, {title}, {url}; licensed under "
                "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)."
            )
            conn.execute(
                "INSERT INTO knowledge_docs VALUES(?,?,?,?,?,?,?,?)",
                (
                    "99", title, url, body, attribution, "CC BY-SA 4.0",
                    "99999", "0" * 64,
                ),
            )
            result = search_knowledge_index(conn, "태양계 행성")
            self.assertEqual(result["status"], "document_metadata_invalid")
            self.assertEqual(result["documents"], [])
            self.assertEqual(result["rejected_source_rows"], 1)
            self.assertEqual(result["rejected_integrity_rows"], 1)
        finally:
            conn.close()

    def test_index_search_reports_database_failure(self):
        conn = sqlite3.connect(":memory:")
        try:
            result = search_knowledge_index(conn, "태양계")
            self.assertEqual(result["status"], "search_unavailable")
            self.assertEqual(result["documents"], [])
        finally:
            conn.close()

    def test_index_search_reports_empty_query_without_database_access(self):
        conn = sqlite3.connect(":memory:")
        try:
            result = search_knowledge_index(conn, "무엇인가요?")
            self.assertEqual(result, {
                "documents": [], "status": "empty_query", "used_fallback": False
            })
        finally:
            conn.close()

    def test_domain_alias_expansion_cannot_exceed_query_term_limit(self):
        from knowledge_retrieval import MAX_QUERY_TERMS

        base_terms = [f"항목{chr(0xAC00 + index)}점" for index in range(MAX_QUERY_TERMS - 1)]
        base_terms[0] = "화학자"
        terms = query_terms(" ".join(base_terms))
        self.assertLessEqual(len(terms), MAX_QUERY_TERMS)
        self.assertIn("화학", terms)

    def test_alias_expansion_respects_full_query_term_budget(self):
        from knowledge_retrieval import MAX_QUERY_TERMS

        base_terms = [f"항목{chr(0xAC00 + index)}점" for index in range(MAX_QUERY_TERMS)]
        base_terms[0] = "화학자"
        terms = query_terms(" ".join(base_terms))
        self.assertEqual(len(terms), MAX_QUERY_TERMS)
        self.assertNotIn("화학", terms)

    def test_multiple_domain_aliases_share_one_global_term_budget(self):
        from knowledge_retrieval import MAX_QUERY_TERMS

        filler = [f"항목{chr(0xAC00 + index)}점" for index in range(56)]
        triggers = ["화학자", "연속", "신호", "변화율", "누적량", "입자", "물리학"]
        terms = query_terms(" ".join(filler + triggers))
        self.assertEqual(len(terms), MAX_QUERY_TERMS)
        self.assertIn("화학", terms)
        self.assertNotIn("아날로그", terms)
        self.assertNotIn("미적분학", terms)
        self.assertNotIn("입자물리학", terms)

    def test_rank_documents_rejects_candidate_batches_over_the_hard_limit(self):
        from knowledge_retrieval import MAX_RANK_CANDIDATES

        rows = [
            {"title": "문서", "text": "검색 내용", "url": f"https://example.test/{i}"}
            for i in range(MAX_RANK_CANDIDATES + 1)
        ]
        with self.assertRaisesRegex(ValueError, "최대"):
            rank_documents("검색 내용", rows)

    def test_rank_documents_rejects_oversized_document_fields(self):
        from knowledge_retrieval import (
            MAX_DOCUMENT_BODY_CHARS,
            MAX_DOCUMENT_TITLE_CHARS,
            MAX_DOCUMENT_URL_CHARS,
        )

        oversized_rows = [
            {"title": "제목" * (MAX_DOCUMENT_TITLE_CHARS // 2 + 1),
             "text": "검색 내용", "url": "https://example.test/doc"},
            {"title": "문서", "text": "가" * (MAX_DOCUMENT_BODY_CHARS + 1),
             "url": "https://example.test/doc"},
            {"title": "문서", "text": "검색 내용",
             "url": "https://example.test/" + "a" * MAX_DOCUMENT_URL_CHARS},
        ]
        for row in oversized_rows:
            with self.subTest(field_lengths={key: len(value) for key, value in row.items()}):
                with self.assertRaisesRegex(ValueError, "허용 길이"):
                    rank_documents("검색 내용", [row])

    def test_rank_documents_accepts_documents_at_configured_size_limits(self):
        from knowledge_retrieval import (
            MAX_DOCUMENT_BODY_CHARS,
            MAX_DOCUMENT_TITLE_CHARS,
            MAX_DOCUMENT_URL_CHARS,
        )

        row = {
            "title": "검색" + "가" * (MAX_DOCUMENT_TITLE_CHARS - len("검색")),
            "text": "나" * MAX_DOCUMENT_BODY_CHARS,
            "url": "https://example.test/" + "a" * (MAX_DOCUMENT_URL_CHARS - len("https://example.test/")),
        }
        # A single title hit is sufficient evidence; the test only checks that
        # exact configured boundaries are accepted rather than rejected.
        ranked = rank_documents("검색", [row], min_score=0.0)
        self.assertEqual(len(ranked), 1)


if __name__ == "__main__":
    unittest.main()
