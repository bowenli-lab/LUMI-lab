from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': "4CR LNP to Cell - Round Bottom x2",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca',
    'description': 'Opentrons liquid dispensing protocol to add LNPs to 2 cell lines, both in round-bottom plates',
    'apiLevel': '2.12'
}

# amounts
lnp_amount = 2

# plate locations
lnp_plate_1_location = 4
lnp_plate_2_location = 1

cell_plate_1a_location = 5
cell_plate_1b_location = 6
cell_plate_2a_location = 2
cell_plate_2b_location = 3

# tip box locations
tip_rack_1_location = 7
tip_rack_2_location = 8
tip_rack_3_location = 10
tip_rack_4_location = 11


def run(protocol: protocol_api.ProtocolContext):
    # load labware
    # load tipracks
    tiprack_1 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_1_location, label='Tip_Rack_1')
    tiprack_2 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_2_location, label='Tip_Rack_2')
    tiprack_3 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_3_location, label='Tip_Rack_3')
    tiprack_4 = protocol.load_labware('opentrons_96_tiprack_20ul', location=tip_rack_4_location, label='Tip_Rack_4')
    # load pipettes
    p20_multi_pipette = protocol.load_instrument('p20_multi_gen2', mount='left',
                                                 tip_racks=[tiprack_1, tiprack_2, tiprack_3, tiprack_4])

    # load lnp plates
    lnp_plate_1 = protocol.load_labware('servicebio_96_wellplate_200ul', location=lnp_plate_1_location,
                                        label='LNP_Plate_1')
    lnp_plate_2 = protocol.load_labware('servicebio_96_wellplate_200ul', location=lnp_plate_2_location,
                                        label='LNP_Plate_2')

    # load cell plates
    cell_plate_1a = protocol.load_labware('biomartubottom_96_wellplate_200ul', location=cell_plate_1a_location,
                                          label='Cell_Plate_1a')
    cell_plate_1b = protocol.load_labware('biomartubottom_96_wellplate_200ul', location=cell_plate_1b_location,
                                          label='Cell_Plate_1b')
    cell_plate_2a = protocol.load_labware('biomartubottom_96_wellplate_200ul', location=cell_plate_2a_location,
                                          label='Cell_Plate_2a')
    cell_plate_2b = protocol.load_labware('biomartubottom_96_wellplate_200ul', location=cell_plate_2b_location,
                                          label='Cell_Plate_2b')

    # transfer lnp to cell plates
    p20_multi_pipette.transfer(lnp_amount, lnp_plate_1.wells(), cell_plate_1a.wells(), new_tip='always')
    p20_multi_pipette.transfer(lnp_amount, lnp_plate_1.wells(), cell_plate_1b.wells(), new_tip='always')
    p20_multi_pipette.transfer(lnp_amount, lnp_plate_2.wells(), cell_plate_2a.wells(), new_tip='always')
    p20_multi_pipette.transfer(lnp_amount, lnp_plate_2.wells(), cell_plate_2b.wells(), new_tip='always')


    # for col in range(1, 12):
    #     p20_multi_pipette.distribute(lnp_amount, lnp_plate_1.columns_by_index()[str(col)],
    #                                  [cell_plate_1a.columns_by_index()[str(col)],
    #                                   cell_plate_1b.columns_by_index()[str(col)]], new_tip="always", disposal_volume=0)
    #     p20_multi_pipette.distribute(lnp_amount, lnp_plate_2.columns_by_index()[str(col)],
    #                                  [cell_plate_2a.columns_by_index()[str(col)],
    #                                   cell_plate_2b.columns_by_index()[str(col)]], new_tip="always", disposal_volume=0)
