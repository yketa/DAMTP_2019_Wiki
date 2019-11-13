all: dir
	@latex figure
	@dvisvgm figure
	@dvips figure -o figure.eps
	@mv `ls figure* | grep -v tex | grep -v svg | grep -v eps` aux

dir:
	@mkdir -p aux

clean:
	@rm -f `ls figure* | grep -v tex | grep -v svg | grep -v eps`
	@rm -rf aux

mrproper: clean
	@rm -f figure.svg
	@rm -f figure.eps
