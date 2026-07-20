DECKS := $(wildcard [0-9][0-9][0-9]-*)

.PHONY: all clean $(DECKS)

all: $(DECKS)

$(DECKS):
	$(MAKE) -C $@ all

clean:
	@for d in $(DECKS); do $(MAKE) -C $$d clean; done
