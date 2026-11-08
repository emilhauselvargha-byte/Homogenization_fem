.PHONY: test figures quick-figures plots diagrams report clean

test:
	pytest

# all figures from scratch (~30 min on two cores)
figures:
	python scripts/02_ellipse_sweeps.py
	python scripts/03_cell_fields.py
	python scripts/04_homogenization_error.py
	python scripts/05_tensor_convergence.py
	python scripts/06_mesh_figure.py
	python scripts/07_boundary_layer.py

# smoke test: small n lists (a couple of minutes); overwrites figures/ and results/
quick-figures:
	python scripts/02_ellipse_sweeps.py
	python scripts/03_cell_fields.py
	python scripts/04_homogenization_error.py --quick
	python scripts/05_tensor_convergence.py --quick

# redraw the convergence figures from results/*.csv
plots:
	python scripts/04_homogenization_error.py --plot-only
	python scripts/05_tensor_convergence.py --plot-only
	python scripts/07_boundary_layer.py --plot-only

# TikZ boundary-condition diagrams -> figures/*.pdf
diagrams:
	cd figures/diagrams && for f in bc_dns bc_cell boundary_layer; do \
		pdflatex -interaction=nonstopmode $$f.tex >/dev/null && mv $$f.pdf ..; done; \
		rm -f *.aux *.log

report:
	cd report && latexmk -pdf homogenization_report.tex

clean:
	rm -rf output .pytest_cache src/*.egg-info
	cd report && latexmk -C
