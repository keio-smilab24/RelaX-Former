"""失敗例をカウント"""
import json


def main():
    # result_path = "result/id104_rec_hm3d_test.json"
    result_path = "result/id104_rec_mp3d_test.json"
    with open(result_path, "r") as json_file:
        result = json.load(json_file)

    # num_fail_targ = 0
    # num_fail_rec = 0
    num_fail = 0

    for env in result.values():
        for idx in range(0, len(env), 2):
        # for sample in env:
            # if float(sample["mrr"]) < 10.0:
            #     if sample["instruction"].startswith("<target>"):
            #         num_fail_targ += 1
            #     else:
            #         num_fail_rec += 1
            if float(env[idx]["mrr"]) < 10.0 and float(env[idx+1]["mrr"]) < 10.0:
                num_fail += 1
                # print(env[idx]["instruction_id"])
                # print(env[idx]["instruction"])

    # print(f"num_fail_targ: {num_fail_targ}")
    # print(f"num_fail_rec: {num_fail_rec}")
    print(f"num_fail: {num_fail}")

    # hm3d < 10.0
    # num_fail_targ: 62
    # num_fail_rec: 61
    # mp3d < 10.0
    # num_fail_targ: 112
    # num_fail_rec: 115

    # hm3d: targ < 10.0 and rec < 10.0
    # num_fail: 21
    # 3t8DB4Uzvkt_target_000198_to_dest_000243_571211
    # 7GAhQPFzMot_target_000117_to_dest_000039_921919
    # y9hTuugGdiq_target_000432_to_dest_000311_217963
    # T6nG3E2Uui9_target_001713_to_dest_001371_850650
    # T6nG3E2Uui9_target_001716_to_dest_001383_608261
    # T6nG3E2Uui9_target_001733_to_dest_001824_434655
    # bCPU9suPUw9_target_000936_to_dest_000644_482198
    # LNg5mXe1BDj_target_000027_to_dest_000007_352507
    # Nfvxx8J5NCo_target_000018_to_dest_000044_571211
    # a8BtkwhxdRV_target_000004_to_dest_000077_571211
    # a8BtkwhxdRV_target_000066_to_dest_000073_800469
    # q3hn1WQ12rz_target_001663_to_dest_001034_878274
    # QaLdnwvtxbs_target_000131_to_dest_000079_147516
    # XB4GS9ShBRE_target_000711_to_dest_000667_647367
    # eF36g7L6Z9M_target_000099_to_dest_000366_442434
    # eF36g7L6Z9M_target_000101_to_dest_000385_571211
    # vBMLrTe4uLA_target_000306_to_dest_000329_984013
    # vBMLrTe4uLA_target_000319_to_dest_000136_921919
    # rJhMRvNn4DS_target_000077_to_dest_000159_891587
    # rJhMRvNn4DS_target_000266_to_dest_000359_352507
    # rJhMRvNn4DS_target_000273_to_dest_000351_104675

    # mp3d: targ < 10.0 and rec < 10.0
    # num_fail: 77
    # QUCTc6BB5sX_target_158_bbox_267_to_dest_007
    # QUCTc6BB5sX_target_196_bbox_304_to_dest_024
    # QUCTc6BB5sX_target_196_bbox_304_to_dest_024
    # QUCTc6BB5sX_target_196_bbox_304_to_dest_024
    # QUCTc6BB5sX_target_196_bbox_304_to_dest_024
    # QUCTc6BB5sX_target_215_bbox_172_to_dest_003
    # QUCTc6BB5sX_target_215_bbox_172_to_dest_003
    # QUCTc6BB5sX_target_215_bbox_172_to_dest_003
    # QUCTc6BB5sX_target_215_bbox_172_to_dest_003
    # QUCTc6BB5sX_target_230_bbox_174_to_dest_005
    # QUCTc6BB5sX_target_230_bbox_174_to_dest_005
    # QUCTc6BB5sX_target_230_bbox_174_to_dest_005
    # QUCTc6BB5sX_target_230_bbox_174_to_dest_005
    # QUCTc6BB5sX_target_230_bbox_174_to_dest_005
    # QUCTc6BB5sX_target_254_bbox_204_to_dest_021
    # QUCTc6BB5sX_target_254_bbox_204_to_dest_021
    # QUCTc6BB5sX_target_254_bbox_204_to_dest_021
    # QUCTc6BB5sX_target_254_bbox_209_to_dest_025
    # QUCTc6BB5sX_target_254_bbox_209_to_dest_025
    # QUCTc6BB5sX_target_254_bbox_209_to_dest_025
    # QUCTc6BB5sX_target_264_bbox_208_to_dest_023
    # QUCTc6BB5sX_target_264_bbox_208_to_dest_023
    # QUCTc6BB5sX_target_264_bbox_208_to_dest_023
    # QUCTc6BB5sX_target_312_bbox_331_to_dest_010
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_350_bbox_181_to_dest_009
    # QUCTc6BB5sX_target_392_bbox_309_to_dest_026
    # QUCTc6BB5sX_target_549_bbox_288_to_dest_010
    # QUCTc6BB5sX_target_557_bbox_292_to_dest_014
    # QUCTc6BB5sX_target_557_bbox_292_to_dest_014
    # QUCTc6BB5sX_target_614_bbox_327_to_dest_006
    # QUCTc6BB5sX_target_6_bbox_328_to_dest_008
    # QUCTc6BB5sX_target_765_bbox_296_to_dest_018
    # X7HyMhZNoso_target_71_bbox_422_to_dest_011
    # ---
    # <target> Pick up the vase on the black shelf and put it on the brown square table.
    # <target> Pick up the picture on the wall and put it on the closet.
    # <target> Pick up a flame on the wall and put is aside of the blown cabinet near the window.
    # <target> Pick up the painting on the wall and put it on the low shelf.
    # <target> Place the painting hanging on the wall under the light on the black platform in the right corner.
    # <target> Please put the ball on the desk.
    # <target> Pick up a ball on the wall and put it on the board in the center of the room.
    # <target> put the exercise ball on the floor and the put it on the cooking table in the kitchen.
    # <target> Pick up the balance ball and put it on top of the work table on the middle of the room. 
    # <target> Pick up a picture and put it on the kitchen table.
    # <target> Pick up a picture on the wall and put it on the table near the coffee maker.
    # <target> Pick up a picture frame on the wall next to the brown tall cabinet and put it on the island in the kitchen.
    # <target> Pick up the painting in the center of the room and place it on a light brown table.
    # <target> Pick up a picture on the wall and put it on a light brown table.
    # <target> Put the red towel on the round wooden table
    # <target> Put the red towel in the kitchen on the wooden table.
    # <target> Pick up the red towel hung on the table and put it on the brown table.
    # <target> Pick up the red towel and put it on the brown shelf.
    # <target> Pick up a red towel on railing and put it on the wooden table with a potted plant, a wooden frame and a plastic bag on it.
    # <target> Pick up the red towl beside the kitchen table and put it on the wooden table.
    # <target> Pick up a white towel locating left of you and put it on the dark brown chest.
    # <target> Pick up a towel its first from left and put it on vintage desk 
    # <target> Pick up the lest side of the white towel and put it on the brown table by wall.
    # <target> Bring the wooden chair in the wine cellar to the wooden table with the lamp on it.
    # <target> Pick up a small picture frame on the upper left and put it on the white table .
    # <target> Pick up the photo frame on the wall and place it on the table being placed behind the sofa.
    # <target> Pick up the picture in the left top and put it on the white table.
    # <target> Take the picture frame hanging on the wall and put it on the display shelf in the living room.
    # <target> Bring the picture on the wall to the fireplace.
    # <target> Pick up the picture on the upper left of the wall and put it on the white table by the marble wall in the sitting room.
    # <target> Pick up a top left pcture and put it on the table behind sofas.
    # <target> Pick up an picture hung on the wall  and put it on the  table behind the sofas.
    # <target> Pick up a top left brown picture frame and put it on white side table with black iron leg.
    # <target> Taking the picture hanging on the wall and placing it on the white shelf.
    # <target> Take the highest frame on the left side of the wall and put it on the shelf between two chairs. 
    # <target> Move the frames on the wall behind the sofa.
    # <target> Pick up the paintings in the wall and put it on the cabinet in the living room
    # <target> Pick up the upper left picture on the wall and put it on the white shelf back of sofas.
    # <target> Move the upper left photo frame on the wall on top of the fireplace
    # <target> Pick up the upper right picture and put it on the white table.
    # <target> Remove one of the paintings from the wall and put it on the while decorated table.
    # <target> Take the left top picture out of the wall and put it on the white shlf behind the arm chairs.
    # <target> Place the top left painting on the wall on the white table.
    # <target> Pick up the photo frame on the top and on left hand side on the wall, put it on top of the console table.
    # <target> Pick up the picture hung on the wall and put it on the table at the wall.
    # <target> Take a picture from the wall and place it on the shelf.
    # <target> Pick up a painting and place it on a white table
    # <target> Put the painting on the wall on the white table.
    # <target> Remove a picture on the wall and put it on the white round stand near the two chairs.
    # <target> Pick up the picture on the wall and put it on the living board.
    # <target> Pick up a small picture frame on the upper left and put it on the white table between the lights.
    # <target> Take the small frame wich is right from the left from the wall and put it in the marbled type wall mount table in the living room.
    # <target> Pick up the picture hungging on the upper left wall and put it on the shelf behind chairs.
    # <target> Pick up the small framed picture top on the left of the wall and put it on the white marble top table by the wall in the sitting room.
    # <target> Pick up the black photo frame on the wall left of the big one and put it on the fire base.
    # <target> Please put the small picture on the white table.
    # <target> Pick up the photo from very left and top then put it on the uniq white desk
    # <target> Pick up a picture on the wall and put it on the board behind the sofa.
    # <target> Pick up the picture from the wall and put it on the decorative shelf.
    # <target> Take one of the pictures off the wall and put it on the chest.
    # <target> Move one of the paintings on the white wall onto the ledge behind the sofa.
    # <target> Move the upper left photo frame on the wall to the fire place
    # <target> Pick up the picture on the wall and put it on the table lavishly decorated.
    # <target> Pick up the picture frame displayed in the upper left corner and put it on the shelf next to the lamps.
    # <target> Pick up a picture on the top left of the wall and put it on a white table.
    # <target> Pick up the picture on a wall and put it down on a table.
    # <target> Pick up the left upper picture and put it on table between two light stand.
    # <target>  Pick up a picture hanged on the left upper on the wall in the hallway and put it on the white half-round table next to the wall in the living room.
    # <target> Pick up the painting on the upper left of the wall and put it on the side table with arabesque pattern.
    # <target> Remove the curtain from the window and put it on the white table with two vases fulled with some liquid.
    # <target> Place the front frame on the right wall on the wooden shelf.
    # <target> Take the telephone plug off and then move the entire telephone from the counter to the white table beside the wall. 
    # <target> Remove the telephone set on the stand and put it on the white wodden table by the wall.
    # <target> The target object (the painting upper left) is too large to put on the designated destination (small table between the lamps).
    # <target> Take the figurine on the display shelf and put it on the wall shelf.
    # <target> Pick up the big vase with white flowers and put it on the bathroom sink.
    # <target> Pick up a dispenser and put it on the table alnong side on the wall in the livingroom.

    # hm3d: targ < 20.0 and rec < 20.0
    # num_fail: 59
    # mp3d: targ < 20.0 and rec < 20.0
    # num_fail: 109

# poetry run python src/analyze_error.py
if __name__ == "__main__":
    main()
