# Rename files to make it easier to evaluate them using my code.
for file in ./*; do
    renamed=$(echo $file | sed 's!\(.*\.ckpt\).*-[0-9]*\(\..*\)!\1\2!')
    mv $file $renamed
done
