from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.defects.utils import ChargeDensityAnalyzer
from pymatgen.analysis.defects.utils import logger as defect_utils_logger
from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import VoronoiNN
from copy import copy
import numpy as np
import operator
import pandas as pd
import networkx as nx
import logging


# logger.basicConfig(filename='defect_utils.log')
defect_utils_logger.setLevel(logging.WARNING)
alpha = list('abcdefghijklmnopqrstuvwxyz')
sm = StructureMatcher()


def generic_groupby(list_in, comp=operator.eq, lab_num=True):
    '''
    Group a list of unhasable objects
    Return a list of labels that represent which group the entry is in
    '''
    list_out = ['TODO'] * len(list_in)
    cnt = 0
    for i1, ls1 in enumerate(list_out):
        if ls1 != 'TODO':
            continue

        if not lab_num:
            list_out[i1] = alpha[cnt]
        else:
            list_out[i1] = cnt
        cnt += 1
        for i2, ls2 in enumerate(list_out):
            if comp(list_in[i1], list_in[i2]):
                list_out[i2] = list_out[i1]
    return list_out


class intsitecomp(ChargeDensityAnalyzer):
    def __init__(self, chgcar, wion='Li'):
        self.working_ion = wion
        super().__init__(chgcar)
        self.get_local_extrema()
        if len(self._extrema_df) > 1:
            self.cluster_nodes(tol=0.6)
        self.sort_sites_by_integrated_chg()
        # self._extrema_df[['site_label']] = self._extrema_df[['site_label']].astype(str)
        # mask_not_dense = np.array(self._extrema_df.avg_charge_den < 0.5)
        # self._extrema_df = self._extrema_df.iloc[mask_not_dense]
        inserted_structs = []
        for itr, li_site in self._extrema_df.iterrows():
            tmp_struct = chgcar.structure.copy()
            li_site = self._extrema_df.iloc[itr]
            tmp_struct.insert(-1, self.working_ion, [li_site['a'],li_site['b'],li_site['c']], properties = {})
            tmp_struct.sort()
            inserted_structs.append(tmp_struct)
        self._extrema_df['inserted_struct'] = inserted_structs
        
    def get_labels(self):
        site_labels = generic_groupby(self._extrema_df.inserted_struct, comp=sm.fit, lab_num=False)
        self._extrema_df['site_label'] = site_labels
        # generate the structure with only Li atoms for NN analysis
        self.allsites_struct = Structure(self.structure.lattice , np.repeat(self.working_ion, len(self._extrema_df)),
                             self._extrema_df[['a', 'b', 'c']].values, 
                             site_properties= {'label' : self._extrema_df[['site_label']].values.flatten()})
        # iterate and make sure that the sites in the allsites_struct are in the same order as the _extrema_df
        self.get_graph()

    def get_graph(self):
        # Generate the graph edges between these sites
        self.gt = StructureGraph.with_local_env_strategy(self.allsites_struct, VoronoiNN())
        self.gt.set_node_attributes()


    def compare_edges(self, edge1, edge2):
        # 
        p0=nx.get_node_attributes(self.gt.graph, 'properties')[edge1[0]]['label']
        p1=nx.get_node_attributes(self.gt.graph, 'properties')[edge1[1]]['label']
        pp0=nx.get_node_attributes(self.gt.graph, 'properties')[edge2[0]]['label']
        pp1=nx.get_node_attributes(self.gt.graph, 'properties')[edge2[1]]['label']
        #print(edge1, '{}->{}'.format(p0, p1), '{}->{}'.format(pp0, pp1), edge2)
        temp_struct1 = self._extrema_df.iloc[edge1[0]]['inserted_struct'].copy()
        new_site = self._extrema_df[['a', 'b', 'c']].values[edge1[1]]
        #print(new_site)
        temp_struct1.insert(0, self.working_ion, new_site, properties = {})
        temp_struct1.sort()

        temp_struct2 = self._extrema_df.iloc[edge2[0]]['inserted_struct'].copy()
        new_site = self._extrema_df[['a', 'b', 'c']].values[edge2[1]]
        #print(new_site)
        temp_struct2.insert(0, self.working_ion, new_site, properties = {})
        temp_struct2.sort()
        #print(sm.fit(temp_struct1, temp_struct2))
        return sm.fit(temp_struct1, temp_struct2)

    def get_edges_labels(self, mask_file=None):
        pos_list_0=[]
        pos_list_1=[]
        to_jimage=[]
        for u, v, k, d in self.gt.graph.edges(keys=True, data=True):
            pos_list_0.append(self._extrema_df[['a', 'b', 'c']].values[u])
            to_jimage.append(d['to_jimage'])
            pos_list_1.append(self._extrema_df[['a', 'b', 'c']].values[v] + np.array(d['to_jimage']))
        pos_list_0= np.array(pos_list_0)                     
        pos_list_1= np.array(pos_list_1)                     
        self._edgelist = pd.DataFrame.from_dict({'edge_tuple' : list(self.gt.graph.edges())})
        edge_lab = generic_groupby(self._edgelist['edge_tuple'], comp = self.compare_edges)
        self._edgelist['edge_label'] = edge_lab
        self._edgelist['to_jimage'] = to_jimage
        self._edgelist['pos0x'] = pos_list_0[:,0]
        self._edgelist['pos0y'] = pos_list_0[:,1]
        self._edgelist['pos0z'] = pos_list_0[:,2]
        self._edgelist['pos1x'] = pos_list_1[:,0]
        self._edgelist['pos1y'] = pos_list_1[:,1]
        self._edgelist['pos1z'] = pos_list_1[:,2]
        # write the image
        self.unique_edges = self._edgelist.drop_duplicates('edge_label', keep='first')


        # set up the grid
        aa = np.linspace(0, 1, len(self.chgcar.get_axis_grid(0)),
                         endpoint=False)
        bb = np.linspace(0, 1, len(self.chgcar.get_axis_grid(1)),
                         endpoint=False)
        cc = np.linspace(0, 1, len(self.chgcar.get_axis_grid(2)),
                         endpoint=False)
        AA, BB, CC = np.meshgrid(aa, bb, cc, indexing='ij')
        fcoords = np.vstack([AA.flatten(), BB.flatten(), CC.flatten()]).T

        IMA, IMB, IMC = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')
        images = np.vstack([IMA.flatten(), IMB.flatten(), IMC.flatten()]).T

        # get the charge density masks for each hop (for plotting and sanity check purposes)
        idx_pbc_mask = np.zeros_like(AA)
        surf_idx=0
        total_chg=[]
        if mask_file:
            mask_out = copy(self.chgcar)
            mask_out.data['total'] = np.zeros_like(AA)

        for _, row in self.unique_edges.iterrows():
            pbc_mask = np.zeros_like(AA).flatten()
            e0 = row[['pos0x', 'pos0y', 'pos0z']].astype('float64').values
            e1 = row[['pos1x', 'pos1y', 'pos1z']].astype('float64').values

            cart_e0 = np.dot(e0, self.chgcar.structure.lattice.matrix)
            cart_e1 = np.dot(e1, self.chgcar.structure.lattice.matrix)
            pbc_mask = np.zeros_like(AA,dtype=bool).flatten()
            for img in images:
                grid_pos = np.dot(fcoords + img, self.chgcar.structure.lattice.matrix)
                proj_on_line = np.dot(grid_pos - cart_e0, cart_e1 - cart_e0) / (np.linalg.norm(cart_e1 - cart_e0))
                dist_to_line = np.linalg.norm(
                    np.cross(grid_pos - cart_e0, cart_e1 - cart_e0) / (np.linalg.norm(cart_e1 - cart_e0)), axis=-1)

                mask = (proj_on_line >= 0) * (proj_on_line < np.linalg.norm(cart_e1 - cart_e0)) * (dist_to_line < 0.5)
                pbc_mask = pbc_mask + mask
            pbc_mask = pbc_mask.reshape(AA.shape)
            if mask_file:
                mask_out.data['total'] = pbc_mask
                mask_out.write_file('{}_{}.vasp'.format(mask_file,row['edge_tuple']))


            total_chg.append(self.chgcar.data['total'][pbc_mask].sum()/self.chgcar.ngridpts)

        self.complete_mask=idx_pbc_mask
        self.unique_edges['chg_total']=total_chg
        #self._edgelist['total_chg_masks'] = total_chg_mask

    
    
    