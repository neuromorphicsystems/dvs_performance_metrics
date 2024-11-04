function [N_diffPair,N_samePair,opposite_pair_fraction] = calc_EventPairs(RateImage)


N_samePair = zeros(size(RateImage));
N_diffPair = N_samePair;
opposite_pair_fraction = N_samePair;
for xi = 1:size(RateImage,1)
    for yi = 1:size(RateImage,2)
        N = length(RateImage{xi,yi});
        if N>1
            event_diff_pair = logical(diff(sign(RateImage{xi,yi}(2:end))));
            N_diffPair(xi,yi) = sum(event_diff_pair);
            N_samePair(xi,yi) = N-N_diffPair(xi,yi)-1;
            opposite_pair_fraction(xi,yi) = N_diffPair(xi,yi)/(N-1);
        end
    end
end
