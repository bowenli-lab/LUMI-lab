from opentrons import protocol_api

# metadata
metadata = {
    'protocolName': "Transfer OneGlo - Flat Bottom Plates",
    'author': 'Kevin <kevinchang.wang@mail.utoronto.ca>',
    'description': 'Opentrons liquid dispensing protocol to transfer OneGlo for flat bottom plates following incubation',
    'apiLevel': '2.12'
}

# amounts
transfer_amount = 105

# labware locations

# tip rack locations
tip_rack_1_location = 7
tip_rack_2_location = 4
tip_rack_3_location = 1

# plate locations
cell_plate_1_location = 8
cell_plate_2_location = 5
cell_plate_3_location = 2

opaque_plate_1_location = 9
opaque_plate_2_location = 6
opaque_plate_3_location = 3


# run

def run(protocol: protocol_api.ProtocolContext):
    # load labware
    # load tip racks
    tip_rack_1 = protocol.load_labware('opentrons_96_tiprack_300ul', location=tip_rack_1_location, label='Tip_Rack_1')
    tip_rack_2 = protocol.load_labware('opentrons_96_tiprack_300ul', location=tip_rack_2_location, label='Tip_Rack_2')
    tip_rack_3 = protocol.load_labware('opentrons_96_tiprack_300ul', location=tip_rack_3_location, label='Tip_Rack_3')

    # load pipette
    p300_multi = protocol.load_instrument('p300_multi_gen2', mount='right',
                                          tip_racks=[tip_rack_1, tip_rack_2, tip_rack_3])

    # load plates
    cell_plate_1 = protocol.load_labware('corning_96_wellplate_360ul_flat', location=cell_plate_1_location,
                                         label='Cell_Plate_1')
    cell_plate_2 = protocol.load_labware('corning_96_wellplate_360ul_flat', location=cell_plate_2_location,
                                         label='Cell_Plate_2')
    cell_plate_3 = protocol.load_labware('corning_96_wellplate_360ul_flat', location=cell_plate_3_location,
                                         label='Cell_Plate_3')

    opaque_plate_1 = protocol.load_labware('corning_96_wellplate_360ul_flat', location=opaque_plate_1_location,
                                           label='Opaque_Plate_1')
    opaque_plate_2 = protocol.load_labware('corning_96_wellplate_360ul_flat', location=opaque_plate_2_location,
                                           label='Opaque_Plate_2')
    opaque_plate_3 = protocol.load_labware('corning_96_wellplate_360ul_flat', location=opaque_plate_3_location,
                                           label='Opaque_Plate_3')
    # transfer
    p300_multi.transfer(transfer_amount, cell_plate_1.wells(), opaque_plate_1.wells(), new_tip='always',
                        mix_before=(3, (transfer_amount * 0.5)))
    p300_multi.transfer(transfer_amount, cell_plate_2.wells(), opaque_plate_2.wells(), new_tip='always',
                        mix_before=(3, (transfer_amount * 0.5)))
    p300_multi.transfer(transfer_amount, cell_plate_3.wells(), opaque_plate_3.wells(), new_tip='always',
                        mix_before=(3, (transfer_amount * 0.5)))
