load iMM904
%initCobraToolbox

model = iMM904;

METS_TO_REMOVE = {'h_c', 'h2o_c', 'co2_c', 'o2_c', 'pi_c', 'atp_c', 'adp_c', 'amp_c',...
    'nad_c', 'nadh_c', 'nadp_c', 'nadph_c', 'coa_c', 'thf_c', '5mthf_c',...
    '5fthf_c', 'methf_c', 'mlthf_c', 'nh4_c', 'cmp_c', 'q8_c', 'q8h2_c',...
    'udp_c', 'udpg_c', 'fad_c', 'fadh2_c', 'ade_c', 'ctp_c', 'gtp_c', 'h2o2_c',...
    'mql8_c', 'mqn8_c', 'na1_c', 'ppi_c', 'ACP_c', 'h_m', 'h2o_m', 'h2o_x',...
    'h_x', 'atp_m', 'ppi_m', 'nad_m', 'coa_x', 'nadh_m', 'amp_m', 'co2_m',...
    'atp_x', 'nadp_m', 'coa_m','nadph_m','h_v','pi_m','h_n','nad_x','h_r',...
    'o2_x','amp_x','adp_m','ppi_x','o2_m','atp_n','h2o2_x','h2o_n','nadh_x',...
    'h_g','q6_m','adp_x','pi_x','utp_c','ACP_m','adp_n','fad_m','gmp_c','q6h2_m',...
    'thf_m','h2o_r','fadh2_m','gdp_c','accoa_c','stcoa_c','hdcoa_c','accoa_x',...
    'tdcoa_c','pmtcoa_c','odecoa_c','accoa_m','h2o_v','nh4_m','hco3_c','so3_c',...
    'hdca_c','hdcea_c','ocdca_c','ttdca_c','ocdcea_c','dca_c','ocdcya','ocdycacoa_c',...
    '12dgr_SC_c','ptd1ino_SC_c','coa_r','nadp_r','nadph_r','nadph_x','pa_SC_c',...
    'hexccoa_c','gdp_g','cmp_m','10fthf_c','cer3_24_c','cer3_26_c','dump_c',...
    'cer1_24_c','cer2_24_c','cer1_26_c','cer2_26_c','cdp_c','malACP_m','gdpmann_c',...
    'pc_SC_c','ddca_coa_c','ump_c','adp_v','atp_v','nadp_x'};

mets_not_removed = METS_TO_REMOVE(~ismember(METS_TO_REMOVE, model.mets));

model = removeMetabolites(model,METS_TO_REMOVE,false);

remove_mito_mets = model.mets(~cellfun(@isempty,strfind(model.mets, '_m')));
remove_extra_mets = model.mets(~cellfun(@isempty,strfind(model.mets, '_e')));
remove_x_mets=model.mets(~cellfun(@isempty,strfind(model.mets, '_x')));
remove_g_mets=model.mets(~cellfun(@isempty,strfind(model.mets, '_g')));
remove_v_mets=model.mets(~cellfun(@isempty,strfind(model.mets, '_v')));
remove_r_mets=model.mets(~cellfun(@isempty,strfind(model.mets, '_r')));
remove_n_mets=model.mets(~cellfun(@isempty,strfind(model.mets, '_n')));

%remove all extracellular mets
model = removeMetabolites(model,remove_extra_mets,false); 

%remove empty reactions
model = removeMetabolites(model,METS_TO_REMOVE,false);
model=removeRxns(model,{'BIOMASS_SC5_notrace'});

%remove reactions without substrate
empty_ids = find(sum(abs(model.S),1) == 0);
model=removeRxns(model,model.rxns(empty_ids));

%make a graph
N = model.S; %met x rxns (Stoichiometrix Matrix), dim = n x m
N = N~= 0; % 1 for edge, 0 no edge
m = size(N,1);
n = size(N,2);
Nnew = [zeros(m,m), N; %incidence matrix for bipartite graph
    N', zeros(n,n)];
G =  graph(Nnew); %convert matrix to graph structure

%remove non-connected moduls from the graph
[G,model] = remove_nonconnected_modules(model,G);

%afterwards check again for unused reactions and remove them
empty_ids = find(sum(abs(model.S),1) == 0);
model=removeRxns(model,model.rxns(empty_ids));

ids = find(sum(abs(model.rxnGeneMat),1)==0);
model.genes(ids) = [];
model.rxnGeneMat(:,ids) = [];

%get data
[~,data_genes] = xlsread('ralser_functional_metabolomics','intracellular_concentration_mM','A2:A4679');
[data] = xlsread('ralser_functional_metabolomics','intracellular_concentration_mM','C2:U4679');
[data_zscored] = xlsread('ralser_functional_metabolomics','Z-score','C2:U4679');

%build graph from stoichiometric matrix
N_final = model.S; %final stoichiometric matrix
Nx = N_final;
Ny = (abs(Nx)~=0)*1;
Ngraph = Ny'*Ny;
Ngraph = (Ngraph ~=0)*1;
Gn = graph(Ngraph); %RAG from N_final for the embedding stuff
dists = distances(Gn); %RxR_dist.xlsx

%%%%data trimming
out = ismember(data_genes, model.genes);
out1 = ismember(model.genes,data_genes);

data_genes(~out) = [];
data(~out,:) = [];
data_zscored(~out,:) = [];

model.genes(~out1) = [];
model.rxnGeneMat(:,~out1) = [];
%sort data
data_new = [];
data_new_z = []
for k = 1:length(model.genes)
    gene = model.genes(k);
    id = find(ismember(data_genes, gene));
  
    data_new(k,:) = data(id,:);
    data_z(k,:)   = data_zscored(id,:);
end  

id_rxn_remove = find(sum(model.rxnGeneMat,2)==0);
model=removeRxns(model,model.rxns(id_rxn_remove));