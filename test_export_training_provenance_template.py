import unittest

from export_training_provenance_template import SOURCE_FIELDS, export_template


class ExportTrainingProvenanceTemplateTests(unittest.TestCase):
    def test_template_preserves_rows_without_fabricating_metadata(self):
        rows = [
            {
                "approved": True,
                "question": "출처를 확인할 질문",
                "answer": "검토할 답변",
                "source": "curated-v1",
            },
            {"approved": False, "question": "보류 질문", "answer": "보류 답변"},
        ]
        template = export_template(rows)
        self.assertEqual(template["records_total"], 1)
        record = template["records"][0]
        self.assertEqual(record["legacy_source_label"], "curated-v1")
        self.assertEqual(set(record["source"]), set(SOURCE_FIELDS))
        self.assertTrue(all(value == "" for value in record["source"].values()))
        self.assertFalse(record["provenance_review"]["approved"])
        self.assertFalse(template.get("automatic_approval", False))

    def test_non_array_input_is_rejected(self):
        with self.assertRaises(ValueError):
            export_template({"bad": "shape"})


if __name__ == "__main__":
    unittest.main()
