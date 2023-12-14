#! /usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$( cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd )


batch_initial=1
batch_max=64
output_initial=1
output_max=512

for ((batch=batch_initial; batch<=batch_max; batch=batch*2))
do
    for ((output=output_initial; output<=output_max; output=output*2))
    do
        output_file="${ROOT_DIR}/eval/13b-threads/manifold_llama-"$batch"-"$output".log"
        ${ROOT_DIR}/examples/llama/run_llama.sh ${batch} ${output} 4 > ${output_file}
        echo "Done $batch $output"
    done
done

echo "Done all the executions"