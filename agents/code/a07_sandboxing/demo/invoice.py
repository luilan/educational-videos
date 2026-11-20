"""Invoices for a small print shop: line items, discounts, VAT and a plain-text rendering."""
from dataclasses import dataclass, field
from datetime import date

VAT_RATE = 0.21               # standard rate
REDUCED_VAT_RATE = 0.10       # books and printed matter
CURRENCY = "EUR"


@dataclass
class Item:
    description: str
    unit_price: float         # before VAT
    quantity: int = 1
    reduced_vat: bool = False

    def net(self):
        return round(self.unit_price * self.quantity, 2)

    def vat(self):
        rate = REDUCED_VAT_RATE if self.reduced_vat else VAT_RATE
        return round(self.net() * rate, 2)


@dataclass
class Customer:
    name: str
    vat_id: str = ""
    loyalty_years: int = 0


@dataclass
class Invoice:
    number: str
    customer: Customer
    issued: date = field(default_factory=date.today)
    items: list = field(default_factory=list)

    def add(self, description, unit_price, quantity=1, reduced_vat=False):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if unit_price < 0:
            raise ValueError("unit price cannot be negative")
        self.items.append(Item(description, unit_price, quantity, reduced_vat))
        return self

    def subtotal(self):
        return round(sum(i.net() for i in self.items), 2)

    def discount_rate(self):
        """5% for every full 2 years as a customer, at most 15%; another 5% on orders over 500 net."""
        rate = min(0.05 * (self.customer.loyalty_years // 2), 0.15)
        if self.subtotal() > 500:
            rate += 0.05
        return rate

    def discount(self):
        return round(self.subtotal() * self.discount_rate(), 2)

    def vat_total(self):
        # VAT is charged on the discounted amounts
        factor = 1 - self.discount_rate()
        return round(sum(i.vat() for i in self.items) * factor, 2)

    def total(self):
        return round(self.subtotal() - self.discount() + self.vat_total(), 2)

    def render(self):
        lines = [f"INVOICE {self.number}", f"Date: {self.issued:%Y-%m-%d}", f"Customer: {self.customer.name}"]
        if self.customer.vat_id:
            lines.append(f"VAT ID: {self.customer.vat_id}")
        lines.append("")
        for i in self.items:
            lines.append(f"{i.quantity:>4} x {i.description:<30} {i.net():>10.2f} {CURRENCY}")
        lines.append("")
        lines.append(f"{'Subtotal':<37} {self.subtotal():>10.2f} {CURRENCY}")
        if self.discount():
            lines.append(f"{'Discount':<37} {-self.discount():>10.2f} {CURRENCY}")
        lines.append(f"{'VAT':<37} {self.vat_total():>10.2f} {CURRENCY}")
        lines.append(f"{'TOTAL':<37} {self.total():>10.2f} {CURRENCY}")
        return "\n".join(lines)


def parse_csv_line(line):
    """'description;unit_price;quantity;reduced' -> keyword arguments for Invoice.add."""
    parts = [p.strip() for p in line.split(";")]
    if len(parts) < 2:
        raise ValueError(f"bad line: {line!r}")
    kwargs = {"description": parts[0], "unit_price": float(parts[1].replace(",", "."))}
    if len(parts) > 2 and parts[2]:
        kwargs["quantity"] = int(parts[2])
    if len(parts) > 3:
        kwargs["reduced_vat"] = parts[3].lower() in ("1", "yes", "true", "y")
    return kwargs


def invoice_from_csv(number, customer, text):
    inv = Invoice(number, customer)
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            inv.add(**parse_csv_line(line))
    return inv


if __name__ == "__main__":
    c = Customer("Imprenta Sol", "ESB12345678", loyalty_years=4)
    inv = invoice_from_csv("2026-0042", c, "Posters A2;12,50;20\nBooklets;3.20;150;yes\n# proof copy is free\n")
    print(inv.render())
