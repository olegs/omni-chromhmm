import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis")))
import summary_plots


def test_state_palette_numbered_states_tab20():
    states = ["1", "2", "10", "11", "12", "E1", "E2", "E10", "E11", "E12"]
    palette = summary_plots.state_palette(states)
    cmap = plt.get_cmap("tab20")

    def expected_hex(num):
        r, g, b = cmap((num - 1) % 20)[:3]
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    for num in [1, 2, 10, 11, 12]:
        assert palette[str(num)] == expected_hex(num)
        assert palette[f"E{num}"] == expected_hex(num)

    # State 10, 11, 12 should not have the same color as State 1
    assert palette["11"] != palette["1"]
    assert palette["12"] != palette["1"]
    assert palette["E11"] != palette["E1"]
    assert palette["E12"] != palette["E1"]
