all: diversity.so

diversity.so: diversity.pyx
	python setup.py build_ext --inplace
	rm -r build

clean:
	rm diversity.so
