from sklearn.utils.estimator_checks import check_estimator
from rogc import ROGC

def test_check_estimator():
    return check_estimator(ROGC())


