from stt_data_with_llm.main import is_valid_transcript


def test_validation():

    assert is_valid_transcript(
        "རྒྱ་ནག་གཞུང་གིས་མཚོ་སྔོན་ཞིང་ཆེན་གོ་ལོག་ཁུལ་བཙུགས་ནས་ལོ་འཁོར་བདུན་ཅུ་འཁོ་བའི་མཛད་སྒོ་འཚོགས་ཡོད་པ་བཞིན།",
        "རྒྱ་ནག་གཞུང་གིས་མཚོ་སྔོན་ཞིང་ཆེན་མགོ་ལོག་ཁུལ་བཙུགས་ནས་ལོ་འཁོར་ ༧༠ འཁོར་བའི་མཛད་སྒོ་འཚོགས་ཡོད་པ་བཞིན།་",
    )
    assert not is_valid_transcript(
        "ཕྱི་ཟླ་བརྒྱད་པའི་ནང་རྒྱ་ནག་གཞུང་གི་མགོ་ལོག་མངའ་ཁུལ་གྱི་ས་གནས་གང་སར་དམ་བསྒྲགས་ཤུགས་ཆེར་ཆེ་ཡོད་པ་དང་།",
        "",
    )
    assert not is_valid_transcript(
        "ལྷག་པར་དགན་སྡེ་ཁག་ཏུ་ཆོས་ཕྱོགས་ཀྱི་བྱེད་སྒོ་ལ་དམ་སྒྲགས་དང་།",
        "ལྷག་པར་དུ་དགོན་སྡེ་ཁག་ཏུ་ཆོས་ཕྱོགས་ཀྱི་བྱེད་སྒོ་ལ་དམ་དྲག་དང་བདེ་འཇགས་ཆེད་ཡིན་སྐོར་བརྗོད་ནས།",
    )


if __name__ == "__main__":
    test_validation()
