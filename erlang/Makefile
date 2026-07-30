ERLC = erlc
EBIN = ebin
SRC = $(wildcard src/*.erl)
TEST = $(wildcard test/*.erl)

.PHONY: all test clean

all:
	@mkdir -p $(EBIN)
	$(ERLC) -o $(EBIN) $(SRC)
	@chmod +x autoresearch
	@echo "Built. Run: ./autoresearch info <model.gguf>"

test: all
	@mkdir -p $(EBIN)
	$(ERLC) -o $(EBIN) $(TEST)
	erl -noshell -pa $(EBIN) -eval \
		'case eunit:test(ar_tests, [verbose]) of ok -> halt(0); _ -> halt(1) end.'

clean:
	rm -f $(EBIN)/*.beam
