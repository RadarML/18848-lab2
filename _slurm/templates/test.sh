#!/bin/bash
#SBATCH --job-name={{name}}.{{version}}.test
#SBATCH --partition={{env.queue}}
#SBATCH --gres={{env.gres_eval}}
#SBATCH --output={{results}}/{{name}}/{{version}}/test.%N.%j.log
#SBATCH --error={{results}}/{{name}}/{{version}}/test.%N.%j.err
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

JAXTYPING_DISABLE=1 COLUMNS=120 uv run evaluate.py \
    {{results}}/{{name}}/{{version}} \
    --data-root {{dataset}} \
    --workers {{env.eval_workers}} \
{%- for x in args %}
    '{{ x }}'{% if not loop.last %} \{% endif %}
{% endfor %}

{%- if write_protect %}
if [ $? -eq 0 ]; then
    chmod -R a=rX {{results}}/{{name}}/{{version}}/eval;
fi
{%- endif %}
