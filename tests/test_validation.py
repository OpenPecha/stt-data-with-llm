from stt_data_with_llm.main import is_valid_transcript


def test_validation():
    assert (
        is_valid_transcript(
            "རྒྱ་ནག་གཞུང་གིས་མཚོ་སྔོན་ཞིང་ཆེན་གོ་ལོག་ཁུལ་བཙུགས་ནས་ལོ་འཁོར་བདུན་ཅུ་འཁོ་བའི་མཛད་སྒོ་འཚོགས་ཡོད་པ་བཞིན།",
            "རྒྱ་ནག་གཞུང་གིས་མཚོ་སྔོན་ཞིང་ཆེན་མགོ་ལོག་ཁུལ་བཙུགས་ནས་ལོ་འཁོར་ ༧༠ འཁོར་བའི་མཛད་སྒོ་འཚོགས་ཡོད་པ་བཞིན།",
        )
        == True  # noqa: E712
    )
    assert (
        is_valid_transcript(
            "ཕྱི་ལོ་ཉིས་སྟོང་ཉི་ཤུ་རྩ་བཞི་ལོའི་ཟླ་བ་བརྒྱད་པའི་ནང་།",
            "ཕྱི་ལོ་ ༢༠༢༤ ཟླ་ ༨ ནང་",
        )
        == False  # noqa: E712
    )


if __name__ == "__main__":
    test_validation()
