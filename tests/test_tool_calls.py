import pytest
from ollama import ChatResponse
import src.glados.Extensions.Tools.call_manager as CallManager
from src.glados.Extensions.Tools.call_manager import QuerySentimentResponse, QueryTypeEnum


@pytest.mark.parametrize("text, expected",
                         [('What is the area of a rectangle with length 5 and width 3?', 15),
                          ('What is the area of a rectangle with length 12 and width 12?', 144),
                          ('What is the area of a rectangle with length five and width three?', 15),
                          ('What is the area of a rectangle with length twenty one and width twelve?', 252)
                          ])
def test_basic_tool_call(text, expected):
    assert CallManager.process_tool_call_response(text).result == expected


@pytest.mark.parametrize("text, expected",
                         [('What is the capitol of france', None),
                          ('Who is nelson mandella.', None),
                          ('how much gold is in fort knox.', None),
                          ])
def test_no_tools(text, expected):
    r = CallManager.process_tool_call_response(text)
    assert r.result is None


@pytest.mark.parametrize("text, expected",
                         [('What do I need for salinity with a current value of thirteen', 1100),
                          ('calculate salt to add with a current salinity of 21', 423),
                          ('My current salinity is twenty one, what do I need to add', 423),
                          ('My current salinity is 21, what do I need to add', 423),
                          ('My current salinity is 21, how much salt do I need to add', 423)
                          ])
def test_tool_call_salinity(text, expected):
    assert CallManager.process_tool_call_response(text).result == expected

@pytest.mark.parametrize("text, tname, sname",
                         [('transcribe the most recent OBS recording. Call it notes with aspen', "notes with aspen", "None"),
                          ('transcribe the OBS recording from the file called whip, name it intro meeting with flexion ', "intro meeting with flexion", "whip")
                          ])
def test_tool_call_obs(text, tname, sname):
    tc = (CallManager.process_tool_call_response(text))
    n = (str(tc.arguments['finalTranscriptName']).replace('_', ' '))
    assert n == tname
    assert str(tc.arguments['sourceFileName']) == sname

@pytest.mark.parametrize("text, expected",
                         [('What do I need for salinity with a current value of thirteen', True),
                          ('What is the area of a rectangle with length 12 and width 12?', True),
                          ('What is the circumference of a circle with diameter twenty four', False),
                          ('What is the capitol of france', False),
                          ('What was the last alkalinity reading?', False),
                          ('Transcribe the last OBS recording?', True),
                          ('Transcribe the last OBS recording and call it meeting with aspen technology?', True)
                          ])
def test_is_tool_call_available(text: str, expected: bool):
    assert bool(CallManager.analyze_request_for_tools(text, "llama3.1")) == expected


@pytest.mark.parametrize("current, expected",
                         [(26, 0),
                          (13, 1100),
                          (0, 2200),
                          (21, 423),
                          (5, 1777),
                          ])
def test_salinity_calculation(current, expected):
    assert CallManager.get_salinity(current) == expected


#@pytest.mark.parametrize("text, expected",
#                         [('What do I need for salinity with a current value of thirteen', QueryTypeEnum.system_tool),
#                          ('What is the capitol of france', QueryTypeEnum.general_query),
#                          ('What is the area of a rectangle with length 12 and width 12?', QueryTypeEnum.system_tool),
#                          ('What is the circumference of a circle with diameter twenty four', QueryTypeEnum.general_query),
#                          ('What was the last alkalinity reading?', QueryTypeEnum.general_query),
#                          ('shut down', QueryTypeEnum.internal_behavior)
#                          ])
#def test_is_tool_call_available(text: str, expected: CallManager.QueryTypeEnum):
#    assert CallManager.categorize_request(text).query_sentiment == expected

