from stats import mean

def test_mean():
    assert mean([2, 4, 6]) == 4

if __name__ == "__main__":
    test_mean()
    print("all tests passed")
