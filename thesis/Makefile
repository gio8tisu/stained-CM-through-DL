
PREFIX=main

all:  ## Compile paper
	pdflatex $(PREFIX)
	makeglossaries $(PREFIX)
	biber $(PREFIX)
	pdflatex $(PREFIX)

clean:  ## Clean output files
	rm $(PREFIX).acn $(PREFIX).acr $(PREFIX).alg $(PREFIX).aux $(PREFIX).bcf $(PREFIX).glsdefs $(PREFIX).ist $(PREFIX).lof $(PREFIX).log $(PREFIX).lot $(PREFIX).out $(PREFIX).run.xml $(PREFIX).toc $(PREFIX).bbl $(PREFIX).blg
