function [G1,model] = remove_nonconnected_modules(model,G)

bins = conncomp(G);
val  = mode(bins);
ids  = bins == val;
G1 = subgraph(G,ids);

Nodes = [model.mets;model.rxns];

%remove reactions from model
idx  = find(~ids);
model=removeRxns(model,Nodes(idx));
end