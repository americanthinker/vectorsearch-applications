def test_template(template):
     {'name': 'content', 'dataType': ['text'], 'indexFilterable': False, 'indexSearchable': True}
     names = [property['name'] for property in template]
     assert 'guest' in names, 'guest property not found in template'
     guest = [property for property in template if property['name'] == 'guest']
     assert guest['dataType'] == ['text'], 'guest property not of type text'
     assert guest['indexFilterable'] == True, 'guest property not filterable'
     assert guest['inddexSearchable'] == True, 'guest property not searchable'
     assert 'summary' in names, 'summary property not found in template'
     summary = [property for property in template if property['name'] == 'summary']
     assert summary['dataType'] == ['text'], 'summary property not of type text'
     assert summary['indexFilterable'] == False, 'summary property was marked as filterable'
     assert summary['inddexSearchable'] == True, 'summary property not searchable'
     return "All tests passed!"