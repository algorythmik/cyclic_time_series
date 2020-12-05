projectname := cyclic_time_series
projectroot := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Data sync vars
datapath-local := $(projectroot)data/raw/
datapath-remote := s3://s3_bucket/$(projectname)/datasets/raw/

nb-server:
	@DATA_DIR=$(projectroot)data PYTHONPATH=$(projectroot) jupyter notebook --notebook-dir $(projectroot)research/notebooks

check-deps:
	@aws --version 1>/dev/null 2>/dev/null || \
	  (echo "You need to install AWS CLI." && exit 1)

data-pull:
	$(eval REALPATH := $(subst $(projectroot),,$(datapath-local)))
	@echo "Populate \"$(REALPATH)\" with \"$(datapath-remote)\".."
	@echo $(datapath-local)
	@rm -rf $(datapath-local)
	@aws s3 sync "$(datapath-remote)" "$(datapath-local)"
