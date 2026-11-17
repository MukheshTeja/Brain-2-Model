import pytest
from scripts.evaluate_released import parse_propositional
from cdb2m.fmri.reasoning_task import problems_to_batch


def test_released_parser_encodes_text_and_checks_label():
    row={"premise1":"If p is cold, then q is warm.","premise2":"p is cold.",
         "conclusion":"q is warm.","trial_type":"modus_ponens_true"}
    p=parse_propositional(row,7)
    assert p.label==1 and p.premises[0][0]=="IFTHEN"
    with pytest.raises(ValueError,match="label mismatch"):
        parse_propositional({**row,"trial_type":"modus_ponens_false"},7)
    neg=parse_propositional({**row,"conclusion":"q is not warm.","trial_type":"modus_ponens_false"},7)
    assert neg.label==0 and neg.conclusion[0]=="ASSERTNOT"
    # Truth labels cannot change the feature encoding.
    original=problems_to_batch([p])["inputs"]
    p.label=0
    assert (original==problems_to_batch([p])["inputs"]).all()


def test_released_parser_reports_unsupported_logic():
    row={"premise1":"p is cold or q is warm.","premise2":"p is not cold.",
         "conclusion":"q is warm.","trial_type":"disjunction_elimination_true"}
    with pytest.raises(ValueError,match="unsupported disjunction"):
        parse_propositional(row,7)
