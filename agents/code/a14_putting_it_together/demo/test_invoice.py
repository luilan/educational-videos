from invoice import Customer, Invoice


def test_total():
    inv = Invoice("T-1", Customer("Test", loyalty_years=2))
    inv.add("Posters", 100.0, 2)            # net 200, VAT 42
    # discount 5% -> 10; VAT on the discounted amount 39.90; total 200 - 10 + 39.90
    assert inv.total() == 229.90, inv.total()


if __name__ == "__main__":
    test_total()
    print("ok")
