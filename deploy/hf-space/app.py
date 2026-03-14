import runpy


ns = runpy.run_path(
    "assignments/assignment-1/app/main.py",
    run_name="hf_space_app",
)
demo = ns["create_app"]()


if __name__ == "__main__":
    demo.launch()
