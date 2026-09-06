#!/bin/bash
#SBATCH --job-name={{name}}.{{version}}.train
#SBATCH --partition={{env.queue}}
#SBATCH --gres={{env.gres}}
#SBATCH --output={{results}}/{{name}}/{{version}}/train.%N.%j.log
#SBATCH --error={{results}}/{{name}}/{{version}}/train.%N.%j.err
{%- if env.cpus is not none %}
#SBATCH --cpus-per-gpu={{env.cpus}}
{%- endif %}
{%- if env.mem is not none %}
#SBATCH --mem-per-gpu={{env.mem}}
{%- endif %}
{%- if env.max_time is not none %}
#SBATCH --time={{env.max_time}}
{%- endif %}
{%- if env.nice is not none %}
#SBATCH --nice={{env.nice}}
{%- endif %}

JAXTYPING_DISABLE=1 COLUMNS=120 uv run train.py \
{%- for x in args %}
    '{{ x }}' \
{%- endfor %}
    meta.dataset={{dataset}} \
    meta.results={{results}} \
    meta.name={{name}} \
    meta.version={{version}} \
    meta.compile=true \
    +environment={{env.name}} || exit $?

uv run nrdk export {{results}}/{{name}}/{{version}};
{%- if write_protect %}
chmod -R a=rX {{results}}/{{name}}/{{version}};
chmod +w {{results}}/{{name}}/{{version}};
{%- endif %}
