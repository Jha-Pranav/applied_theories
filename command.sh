total_size_mb=0
echo "Staged Files:"

git ls-files --stage | while read mode obj stage file; do
    size=$(git cat-file -s $obj)
    size_mb=$(echo "scale=2; $size / (1024 * 1024)" | bc)
    total_size_mb=$(echo "scale=2; $total_size_mb + $size_mb" | bc)
    printf "%6s MB %s\n" "$size_mb" "$file"
done | sort -nr

echo "Total Size: $total_size_mb MB"