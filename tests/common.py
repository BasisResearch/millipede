import torch
from numpy.testing import assert_allclose


def assert_close(actual, expected, atol=1e-7, rtol=0, msg="", equal_nan=False):
    if not msg:
        msg = "{} vs {}".format(actual, expected)
    if type(actual) != type(expected):
        raise AssertionError(
            "cannot compare {} and {}".format(type(actual), type(expected))
        )
    if torch.is_tensor(actual) and torch.is_tensor(expected):
        assert_allclose(actual.data.cpu().numpy(), expected.data.cpu().numpy(),
                        atol=atol, rtol=rtol, equal_nan=equal_nan, err_msg=msg)
    else:
        assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan, err_msg=msg)
