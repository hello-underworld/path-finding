from pymatgen.analysis.defects.utils import ChargeDensityAnalyzer
from pymatgen import Structure
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.analysis.local_env import CrystalNN


class Pathfinder:

    """
    Provides a method to find a path way for Li ions to diffuse to the next periodic spot in a given structure
    """

    def __init__(self, cc):

        """
        Creates a pathfinder.

        Args:
            cc: a Chgcar object that is fully delithiated.
        """
        self.cc = cc

    @staticmethod
    def from_file(filename):
        cc = Chgcar.from_file(filename)
        return Pathfinder(cc)

    @staticmethod
    def from_structure(structure):
        Pathfinder.only_li_struct=structure

    @property
    def prop_struct(self):

        """
        Gets a fully lithiated version of the original structure.

        Returns:
            A structure of the original structure with all possible sites filled with Li and the position index of Li atoms
        """

        CDA = ChargeDensityAnalyzer(self.cc)
        CDA.get_local_extrema()
        CDA.extrema_df
        try:
            CDA.cluster_nodes()
        except:
            pass
        CDA.sort_sites_by_integrated_chg()
        all_sites = CDA._extrema_df

        pos_ints = [u for u in all_sites['avg_charge_den'] if u <= (min(all_sites['avg_charge_den'])*5)]
        num_possible_li = len(pos_ints)
        big_struct = CDA.get_structure_with_nodes()

        ori_struct = self.cc.structure
        dict_info = ori_struct.as_dict()
        dict_info['sites'].extend(big_struct.as_dict()['sites'][len(ori_struct):len(ori_struct) + num_possible_li])

        li_pos = []
        for one_specie in dict_info['sites']:
            if one_specie['label'] == 'X0+':
                one_specie['label'] = 'Li'
                one_specie['species'][0]['element'] = 'Li'
                one_specie['species'][0].pop('oxidation_state')
                li_pos.append(one_specie['abc'])

        fully_li_struct = Structure.from_dict(dict_info)

        return fully_li_struct, li_pos

    @property
    def fully_li_struct(self):
        return self.prop_struct[0]

    @property
    def li_pos(self):
        return self.prop_struct[1]

    @property
    def only_li_struct(self):

        """
        Gets a structure that only contains Li atoms at all possible sites.

        Returns:
            a structure of only Li atoms sitting at all possible sites.
        """

        li_pos = self.li_pos
        only_li_struct = Structure(self.cc.structure.lattice, ['Li']*len(li_pos), li_pos)

        return only_li_struct

    def partially_li_struct(self, index_numbers):

        """
        Generates a structure that is partially lithiated with Li atoms at indicated index numbers.
        :param index_numbers: a list of indices of Li to partially lithiate
        :return: a structure that contains Li at indicated indices
        """

        struct = self.cc.structure.as_dict()
        inserts = self.only_li_struct.as_dict()['sites'][index_numbers]
        struct['sites'].append(inserts)
        return Structure.from_dict(struct)

    def dict_info_nn(self, only_li_struct=None, cutoff_r=5):

        """
        Creates a list of dictionaries recording all neighbor information for all Li in only_li_struct.

        Args:
            only_li_struct: the structure that only contains Li atoms.
            cutoff_r: set a radius such that within this radius the periodic image of itself would count as a near neighbor.

        Returns:
            A list of dictionaries that has all the neighbor information of all possible Li sites.
        """

        NN = CrystalNN(distance_cutoffs=(), x_diff_weight=0.0)
        neighbors_data = []

        if only_li_struct == None:
            only_li_struct = self.only_li_struct

        for i in range(0, len(only_li_struct)):
            info = {}
            info['position'] = [0, 0, 0, i]
            info['neighbors'] = []

            for j in NN.get_nn_info(only_li_struct, i):
                entry = list(j['image'])
                entry.append(j['site_index'])
                info['neighbors'].append(entry)

            if only_li_struct.lattice.a <= cutoff_r:
                peri_self = info['position'][:]
                peri_self[0] = 1
                info['neighbors'].append(peri_self)
            if only_li_struct.lattice.b <= cutoff_r:
                peri_self = info['position'][:]
                peri_self[1] = 1
                info['neighbors'].append(peri_self)
            if only_li_struct.lattice.c <= cutoff_r:
                peri_self = info['position'][:]
                peri_self[2] = 1
                info['neighbors'].append(peri_self)
            neighbors_data.append(info)

        return neighbors_data

    def one_more_step(self, len_n_paths, direction, neighbors_data):

        """
        Takes one more step in the path finding process

        Args:
            len_n_paths: a list that contains all the paths of length n
            direction: indicate the direction of next step (0 for along a vector, 1 for along b direction and 2 for along c direction)
            neighbors_data: the list of dictionaries that contains all the neighbor information

        Returns:
            All paths of length n+1 and all paths that contains the endpoint as the next periodic site of the starting point
        """

        all_paths = []
        next_p_site = [0, 0, 0, 0]
        next_p_site[direction] = 1
        len_n1_paths = []
        for i in len_n_paths:
            for a in range(0, len(neighbors_data[i[-1][-1]]['neighbors'])):
                next_step = neighbors_data[i[-1][-1]]['neighbors'][a][:]
                path = i[:]
                for x in range(0, 3):
                    next_step[x] += path[-1][x]
                if max(next_step[0:3]) <= 1 and min(next_step[0:3]) >= -1:
                    if (next_step not in path[:]) and next_step[direction] >= 0:
                        path.append(next_step)
                        next_p_site[3] = path[0][3]
                        if path[-1] == next_p_site:
                            all_paths.append(path)
                        len_n1_paths.append(path)

        return len_n1_paths, all_paths

    def eligible_paths(self, only_li_struct, neighbors_data, num_steps=None, direction=0):

        """
        Counts all the eligible paths up to the number of steps set in the argument.

        Args: only_li_struct: the structure that only contains Li atoms.
            num_steps: maximum number of steps set to find eligible paths. Default is number of Li atoms in structure.
            neighbors_data: the list of dictionaries that have all neighbors data stored.
            direction: the direction of the next periodic site.

        Returns:
            All eligible paths that lead to the next periodic site in the set direction within max number of steps.

        """

        if num_steps is None:
            num_steps = len(only_li_struct)

        eligible_paths = []
        current_path = []
        for i in range(0, len(only_li_struct)):
            current_path.append([[0, 0, 0, i]]) #create all length 0 paths
        for n in range(0, num_steps):
            [current_path, all_paths] = self.one_more_step(current_path, direction, neighbors_data)
            eligible_paths.extend(all_paths)

        return eligible_paths

    def paths_in_index(self, num_steps=None, direction=0, cutoff_r=5, site_index=None):

        """
        Args:
            num_steps: maximum number of steps for a path
            direction: the direction of diffusion. 0 for along a, 1 for along b, 2 for along c
            cutoff_r: set a radius such that within this radius the periodic image of itself would count as a near neighbor
            site_index: indicate a site index to get paths. Default set to get paths for all sites.

        Return:
            A list of paths to next periodic site indicated by [a,b,c,i] where a,b,c is the image of the position and i is the site index

        """

        only_li_struct = self.only_li_struct

        if num_steps==None:
            num_steps=len(only_li_struct)

        neighbors_data = self.dict_info_nn(only_li_struct,cutoff_r)
        all_paths = self.eligible_paths(only_li_struct, neighbors_data, num_steps, direction)

        if site_index != None:
            all_paths = [u for u in all_paths if u[0][3] == site_index]

        return all_paths

    def li_struct_to(self, fmt, filename):

        """
        Write the structure that only contains possible Li sites to a file

        :param fmt: file format of the output file
        :param filename: name of the output file
        """

        self.only_li_struct.to(fmt=fmt, filename=filename)

    def ori_struct_to(self, fmt, filename):

        """
        Write the structure of the original compound to a file

        :param fmt: file format of the output file
        :param filename: name of the output file
        """

        self.cc.structure.to(fmt=fmt, filename=filename)

    def pop_struct_to(self, fmt, filename):

        """
        Write the structure that is fully populated by Li to a file

        :param fmt: file format of the output file
        :param filename: name of the output file
        """

        self.fully_li_struct.to(fmt=fmt, filename=filename)
