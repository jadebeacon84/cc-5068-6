import os
import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

GUI_ENABLED = os.environ.get("VAD_TEST_GUI") == "1" and bool(os.environ.get("DISPLAY"))


@unittest.skipUnless(GUI_ENABLED, "set VAD_TEST_GUI=1 with a DISPLAY to run GUI smoke")
class GuiSmokeTests(unittest.TestCase):
    def test_window_load_recompute_select_and_export(self):
        import tkinter as tk
        from vad_analyzer.gui import VadApp, EXAMPLE_WAV

        if not EXAMPLE_WAV.exists():
            self.skipTest("example dialogue audio has not been generated")

        root = tk.Tk()
        try:
            app = VadApp(root)
            for _ in range(3):
                root.update_idletasks()
                root.update()

            app.load_example()
            for _ in range(5):
                root.update_idletasks()
                root.update()

            self.assertGreaterEqual(len(app.tree_iv.get_children()), 2)
            self.assertIsNotNone(app.decision)

            first = app.tree_iv.get_children()[0]
            app.tree_iv.selection_set(first)
            app._on_tree_select()
            self.assertIsNotNone(app._selected_interval())

            app.var_enter.set(0.5)
            app._on_param("enter")
            root.update_idletasks(); root.update()

            app._zoom(1.2)
            app._zoom_fit()
            for tab in app.nb.tabs():
                app.nb.select(tab)
                root.update_idletasks(); root.update()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
