from opentrons import protocol_api


# 1. take the cell plate out from incubator into opentron
# 2. In opentron: Transfer cells into a new white 96-well plate
# 3. In opentron: Add one-glow reagent
# 4. wait for 10 mins
# 5. In opentron: take the well into plate reader

# metadata
metadata = {
    "protocolName": "Step 4 Plate Reading",
    "author": "Haotian Cui <subercui@gmail.com>",
    "description": "Transfer cells to new well plate before reading, and add reagent to the well plate.",
}

# requirements
requirements = {"robotType": "OT-2", "apiLevel": "2.15"}


# protocol run function
def run(protocol: protocol_api.ProtocolContext):
    # DECLARE THE EQUIPMENT INFO AND STATUS

    # labware
    tipracks = [protocol.load_labware("opentrons_96_tiprack_300ul", str(4))]
    previous_cell_plate = protocol.load_labware(
        "corning_96_wellplate_360ul_flat", location="1"
    )
    new_cell_plate = protocol.load_labware(
        "corning_96_wellplate_360ul_flat", location="2"
    )
    cold_deepwell = protocol.load_labware(
        "customizedcoolerplate_96_wellplate_1300ul", location="6"
    )

    # pipettes
    right_multi_pipette = protocol.load_instrument(
        "p300_multi_gen2", mount="right", tip_racks=tipracks
    )

    # Get the bottom of the wells
    previous_cell_plate_wells_bottom = [
        well.bottom(-0.4) for well in previous_cell_plate.wells()
    ]

    # ACTION COMMANDS
    # left_pipette.pick_up_tip()
    # left_pipette.aspirate(100, plate["A1"])
    # left_pipette.dispense(100, plate["B2"])
    # left_pipette.drop_tip()
    # transfer cells from previous cell plate to new cell plate
    right_multi_pipette.transfer(
        25,
        cold_deepwell.columns()[5],
        previous_cell_plate_wells_bottom,
        mix_after=(3, 150),
        new_tip="always",
        trash=False,
    )
    # wait 3 mins
    protocol.delay(minutes=3)

    right_multi_pipette.reset_tipracks()

    # add reagent to the new cell plate
    right_multi_pipette.transfer(
        150,
        previous_cell_plate_wells_bottom,
        new_cell_plate.wells(),
        mix_before=(1, 150),
        new_tip="always",
        trash=False,
    )
