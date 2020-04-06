all: dir
	@pdflatex main
	@pdflatex main
	-@biber main || bibtex main
	@pdflatex main
	@pdflatex main
	@mv `ls main.* | grep -v tex | grep -v pdf` aux

dir:
	@mkdir -p aux

clean:
	@rm -f `ls main.* | grep -v tex | grep -v pdf`
	@rm -rf aux

mrproper: clean
	@rm -f *converted-to.pdf
	@rm -f main.pdf

