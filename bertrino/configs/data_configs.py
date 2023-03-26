from dataclasses import dataclass


@dataclass
class GCSPathInfo:
    competition_data_path: str = "gs://kds-30cdda2cbed7e7cafd4b18d260c4bcc42d34fe5d60fbf271ba87e0ed"
    batched_ndi_metadata_path: str = "gs://kds-790e95547401ba8e1c009e78cf3fedd5c334bedfa3b5389682b5a131"
    tfrecords_part_1_path: str = "gs://kds-94b5827854f0ef53320289c5e93dd429d0ec643372fe07b44f89c610/tfrecords/train"
    tfrecords_part_2_path: str = "gs://kds-b0da8a1f7d298a24fa1f0e2c2b770447cdb1ec0b18d3c24e5902e7b5/tfrecords/train"
    tfrecords_part_3_path: str = "gs://kds-3c430cd8691fa79503700b64d12545a253c989ec9a45f913f94bdea6/tfrecords/train"
    tfrecords_part_4_path: str = "gs://kds-8a6a600e0dc79d6134c418ba314135d0a9c4f7bc48278a4e10397b45/tfrecords/train"
    tfrecords_part_5_path: str = "gs://kds-b3e420870616b98f6f51448212fe5155ebe6a0a09387cbf16eb4f18e/tfrecords/train"
    tfrecords_part_6_path: str = "gs://kds-b66289bd2a06fe6a4ac5a886afd49b1b6310f869f0e46f7cf7f0eef5/tfrecords/train"
