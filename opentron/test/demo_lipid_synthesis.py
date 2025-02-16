from opentrons import protocol_api

# Metadata defines the required parameters for the protocol to be uploaded and run on the OT-2.
metadata = {
    "protocolName": "Lipid Synthesis Protocol",
    "author": "Kuan Pang <kuan.pang@mail.utoronto.ca>",
    "description": "Lipid synthesis to a 96-well plate",
}
requirements = {"robotType": "OT-2", "apiLevel": "2.15"}

# targets: List[List[4]]; 96 x 4 list of reagent locations for each 96 targets
targets = [[] for _ in range(96)]

# targets is 96 x 4 list of reagent locations
targets[0] = []  # blank control well 0
targets[1] = []  # blank control for well 1
targets[2] = []  # MC3 control for well 2
targets[3] = []  # MC3 control for well 3

####################
# REPLACE ME WITH THE REST OF THE TARGETS
####################


def run(protocol: protocol_api.ProtocolContext):
    # Define labware
    tiprack_300_list = [
        protocol.load_labware("opentrons_96_tiprack_300ul", str(slot))
        for slot in [3, 5, 6, 9]
    ]

    reagent_plate = protocol.load_labware("thermoscientificnunc_96_wellplate_1300ul", "1")
    destination_plate = protocol.load_labware("servicebio_96_wellplate_200ul",
                                              "2")

    # Define pipette
    p300 = protocol.load_instrument("p300_single", "right",
                                    tip_racks=tiprack_300_list)

    # Adjust based on actual reagent locations
    for i in range(5):
        dest_well = destination_plate.wells()[i]
        # mix at the last step
        for idx, reagent_num in enumerate(targets[i]):
            source_well = reagent_plate.wells()[reagent_num]
            # Transfer reagent
            p300.transfer(
                20,
                source_well,
                dest_well,
                new_tip="always",
                trash=False,
                mix_after=(3, 70) if idx == len(targets[i]) - 1 else None,
            )
