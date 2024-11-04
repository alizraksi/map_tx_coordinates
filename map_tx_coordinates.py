#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import re
import warnings


class CIGARString:
    """ Class representing a CIGAR string describing a genomic mapping. Implemented as a list containing integer-char
        tuples.
    """
    def __init__(self, cigar_str):
        self.cigar_str = cigar_str
        self.cigar = self.__parse_cigar_str(cigar_str)

    @staticmethod
    def __parse_cigar_str(s):
        """ Parse CIGAR string and return list containing integer-char tuples. Valid chars are M, D, I, S, H, =, X.
        For example, the CIGAR str '8M7D6M2I' would return [('8', 'M'), ('7', 'D'), ('6', 'M'), ('2', 'I')] """
        chars_list = [char for char in s if char.isalpha()]
        int_list = re.split('[MDISH=X]', s)
        int_list.pop()     # Remove empty string at the end of list
        return list(zip([int(i) for i in int_list], chars_list))

    def map_coordinate(self, tx_pos: int, mapping_start_pos: int) -> int:
        """ Translate a (0-based) transcript coordinate to a (0 based) genome coordinate. For example, for the mapping
        where mapping_start_pos=3 and the mapping is 8M7D6M2I2M11D7M, the fifth base in TR1 (i.e. TR1:4) maps to genome
        coordinate CHR1:7.

        Key strengths: Runs in O(N) time at worst, where N is the transcript length, and runs faster if there are large
            deletions and insertions. Solution is very efficient in case of long chromosomes, since runtime does
            not depend on chromosome length.

        Weaknesses / limitations:
            * If there are many queries to a single chromosome, this is not the most efficient
              solution, because alignments get re-calculated for each query, instead of storing mapped coordinates.
            * Current implementation handles M (match), D (deletion), and I (insertion) regions only.

        This function was tested for cases such as tx_pos at the edge of a deletion or insertion region,
        tx_pos falling within a region, and tx_pos that is out-of-bounds considering the transcript length

        :param tx_pos: integer representing query transcript coordinate that we are mapping
        :param mapping_start_pos: integer representing the genomic coordinate at which the transcript starts to align
        :return integer representing genomic coordinate that the transcript coordinate maps to
        :raises ValueError: if query transcript coordinate out-of-bounds
        """

        # Check that we have valid input coordinates
        if tx_pos < 0 or tx_pos >= self.get_tx_length():
            raise ValueError(f"Transcript position out of bounds (tx_pos = {tx_pos}, transcript length "
                             f"= {self.get_tx_length()}, CIGAR str = {self.cigar_str})")

        # Initialize position at mapping_start_pos
        curr_pos = mapping_start_pos

        # Note type of alignment region we're in, e.g. match of length 8
        curr_region_length = self.cigar[0][0]
        curr_region_type = self.cigar[0][1]
        region_ix = 0               # keep track of which region we're in, e.g. 8M is region 0
        within_region_ix = 0        # which position within a region curr_pos is at, e.g. pos 0 within the 8M region
        tx_ix = 0                   # which position of the transcript are we currently at
        reached_end_of_region = False

        # Step through transcript until we get to query position, adjusting curr_pos depending on whether we're in
        # a region of match, deletion or insertion
        while tx_ix < tx_pos:

            if curr_region_type == 'M':
                curr_pos += 1
                within_region_ix += 1
                if within_region_ix == (curr_region_length - 1):
                    reached_end_of_region = True
            elif curr_region_type == 'D':
                curr_pos += curr_region_length + 1
                reached_end_of_region = True        # jump over entire region and skip to next region
            elif curr_region_type == 'I':
                # skip ahead along transcript according to the size of insertion but only shift curr_pos up by one
                tx_ix += curr_region_length
                curr_pos += 1
                reached_end_of_region = True

                # Did we skip over our transcript coordinate? If so, warn that we have an invalid entry
                if tx_pos <= tx_ix:
                    warnings.warn(f"Transcript coordinate maps to insertion region and does not have a corresponding "
                                  f"genomic coordinate. Returning genomic coordinate to the right of insertion. "
                                  f"(tx_pos = {tx_pos}, CIGAR str = {self.cigar_str})")

            # If we've reached the end of a region, update current region with whatever region comes next
            if reached_end_of_region:
                region_ix += 1
                if region_ix < len(self.cigar):
                    curr_region_length = self.cigar[region_ix][0]
                    curr_region_type = self.cigar[region_ix][1]
                    within_region_ix = 0
                    reached_end_of_region = False

            tx_ix += 1

        return curr_pos

    def get_tx_length(self):
        """ Calculate transcript length by adding up lengths of match and insertion regions """
        region_lengths = [region[0] for region in self.cigar if region[1] in ['M', 'I']]
        return sum(region_lengths)


def run(transcripts_fn, queries_fn, output_fn):
    """Read input files containing transcript mappings and query transcript coordinates, and output file containing
    coordinates that have been mapped to chromosome coordinates. All coordinates are 0-based.
    Assumptions: input files are correctly formatted.

    :param transcripts_fn: filepath to tab-delimited input file containing list of transcripts and their genomic
        mapping. Columns are [tx_id, chrom_id, mapping_start_pos, CIGAR_str], and the file contains no header.
    :param queries_fn: filepath to tab-delimited input file containing list of queries with transcript coordinates.
        Columns are [tx_id, tx_pos], and the file contains no header.
    :param output_fn: filepath to output file. Contains the following columns: [tx_id, tx_pos, chrom_id, chrom_pos]
    :raises ValueError: if error parsing input file, e.g. unexpected number of columns
    """

    # Read transcripts file
    df_transcripts = pd.read_csv(transcripts_fn, delimiter='\t', index_col=0, header=None,
                              names=['tx_id', 'chrom_id', 'mapping_start_pos', 'CIGAR_str'])

    # Read each line in queries file and output mappings line-by-line
    with open(output_fn, 'w') as out_file:
        with open(queries_fn, 'r') as queries_file:

            for line in queries_file:
                try:
                    tx_id, tx_pos = line.strip().split('\t')
                except ValueError:
                    print(f"Error parsing query file, expecting 2 values, check formatting in line: {line}")
                    raise
                chrom_id, chrom_pos = get_coordinate_mapping(tx_id, int(tx_pos), df_transcripts)
                out_file.write('\t'.join([tx_id, str(tx_pos), chrom_id, str(chrom_pos)]) + '\n')
    print(f'Mappings done, output to {output_fn}')


def get_coordinate_mapping(tx_id: str, tx_pos: int, df_transcripts: pd.DataFrame) -> (str, int):
    """Find query transcript in df_transcripts and return its mapping to genomic coordinates.
    Assumptions: each tx_id maps to a single chromosome, i.e. tx_id occurs at most once in df_transcripts.

    :param tx_id: string containing transcript ID, example 'TR1'
    :param tx_pos: integer containing transcript coordinate (0-based)
    :param df_transcripts: DataFrame containing mappings of all transcripts
    :return tuple containing chromosome ID and integer indicating chromosome coordinate (0-based)
    :raises KeyError: if query transcript not found in df_transcripts
    """

    try:
        tx_mapping = df_transcripts.loc[tx_id]
    except KeyError:
        print(f"Query transcript ID ({tx_id}) not found in transcript mappings DataFrame")
        raise

    cigar = CIGARString(tx_mapping.CIGAR_str)
    chrom_pos = cigar.map_coordinate(tx_pos, tx_mapping.mapping_start_pos)
    return tx_mapping.chrom_id, chrom_pos


def main():
    # command line arguments
    parser = argparse.ArgumentParser(
        description='Translate transcript coordinates to genomic coordinates.')
    parser.add_argument('--transcripts', '-t', required=True,
                        help='Path to file containing list of transcripts and their genomic mapping.')
    parser.add_argument('--queries', '-q', required=True,
                        help='Path to file containing list of queries with transcript coordinates.')
    parser.add_argument('--output', '-o', required=True,
                        help='Output file containing chromosome mapping coordinates.')

    args = parser.parse_args()

    run(transcripts_fn=args.transcripts, queries_fn=args.queries, output_fn=args.output)

    return 0


if __name__ == '__main__':
    main()