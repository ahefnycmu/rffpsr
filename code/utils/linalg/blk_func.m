function Y = blk_func(f, n, blk)
if nargin < 3 || isempty(blk); blk = 1000; end   

x = f(1,1);
num_blocks = ceil(n/blk);

for b = 1:num_blocks
    blk_start = (b-1)*blk+1;
    blk_end = min(n, blk_start + blk);
    Y(:,blk_start:blk_end) = f(blk_start, blk_end);
end
