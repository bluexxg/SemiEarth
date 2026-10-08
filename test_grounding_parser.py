import unittest
from util.grounding_parser import parse_grounding

class GroundingParserTests(unittest.TestCase):
    def parse(self, text):
        return parse_grounding(text, ['Building', 'Road'])

    def test_legacy(self):
        self.assertEqual(self.parse('building: [1, 2, 10, 20]'), {'Building': [(1,2,10,20)]})

    def test_json_fence_and_field_order(self):
        text = '```json\n[{"bbox_2d":[1,2,10,20],"label":"Building"}]\n```'
        self.assertEqual(self.parse(text), {'Building': [(1,2,10,20)]})

    def test_complete_records_before_truncated_tail(self):
        text = '[{"class_name":"Building","bbox_2d":[1,2,10,20]}, {"label":"Road","bbox_2d":[0,0'
        self.assertEqual(self.parse(text), {'Building': [(1,2,10,20)]})

    def test_no_salvage_of_incomplete_object(self):
        self.assertEqual(self.parse('{"label":"Road","bbox_2d":[0,0,10,20]'), {})

    def test_invalid_records(self):
        import json
        records = [
            {'label':'Unknown','bbox_2d':[1,2,3,4]},
            {'label':'Road','bbox_2d':[1,2,1,4]},
            {'label':'Road','bbox_2d':[-1,2,3,4]},
            {'label':'Road','bbox_2d':[True,2,3,4]},
            {'label':'Road','bbox_2d':[1,2,3]},
            {'label':'Road','class_name':'Building','bbox_2d':[1,2,3,4]},
        ]
        self.assertEqual(self.parse(json.dumps(records)), {})

    def test_dedup_and_class_boundary(self):
        self.assertEqual(self.parse('NotBuilding: [1,2,3,4]'), {})
        self.assertEqual(self.parse('Road: [1,2,3,4]\nRoad: [1,2,3,4]'), {'Road':[(1,2,3,4)]})

if __name__ == '__main__':
    unittest.main()
