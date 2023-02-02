#### Final articulation annotation JSON data
```json
{
    "intrinsic": {
        "matrix": "9x1 vector of column major camera intrinsics"
    },
    "extrinsic": {
        "matrix": "16x1 vector of column major camera extrinsics (camera extrinsic)"
    },
    "scan_obb": "scan level 3d bounding box",
    "articulation": [
        {
            "bbox": "Bounding box of the openable part",
            "axis": "aligned axis of the joint in object local common coordinate frame",
            "isClosed": "bool type, whether the part is close or not",
            "origin": "aligned origin of the joint in object local common coordinate frame",
            "pixel_num": "number of pixels of the openable part (this is for further frame filter procedure)",  
            "partId": "part ID",
            "part_label": "part label",
            "rangeMin": "min range of the motion",
            "rangeMax": "max range of the motion", 
            "state": "current state (degree/length) of the corresponding part",
            "type": "articulation type rotation or translation"        
        }
        "..."
    ]
}
```
#### Per frame part and object information JSON data
```json
{
    "object_info": [
        {
            "object_id": "object id",
            "pixel_count": "number of pixels of the object",
            "total_vertex_count": "object vertex number in the scan",
            "vertex_couont": "object vertex number in the frame region"
        },
        "..."
    ],
    "part_info": [
        {
            "part_id": "part_id",
            "pixel_count": "number of pixels of the part",
            "total_vertex_count": "part vertex number in the scan",
            "vertex_couont": "part vertex number in the frame region"
        },
        "..."
    ]
}
```