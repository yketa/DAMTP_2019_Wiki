all: dir
	@pdflatex summary_19_11
	@pdflatex summary_19_11
	@bibtex summary_19_11
	@pdflatex summary_19_11
	@pdflatex summary_19_11
	@mv `ls summary_19_11* | grep -v tex | grep -v pdf` aux

dir:
	@mkdir -p aux

clean:
	@rm -f `ls summary_19_11* | grep -v tex | grep -v pdf`
	@rm -rf aux

mrproper: clean
	@rm -f summary_19_11.pdf

