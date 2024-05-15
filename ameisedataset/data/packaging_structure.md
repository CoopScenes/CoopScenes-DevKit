# Record (t.b.d.)
| Description            | BYTE_LENGTH            |
|------------------------|------------------------|
| MetaInformation_length | 4                      |
| MetaInformation        | variable               |
| Frame_quantity         | 4                      |
| Frame_lengths          | 4 bytes per frame (10) |
| Frames                 | variable times 10      |

### Frame
| Description | BYTE_LENGTH  |
|-------------|--------------|
| Tower       | variable     |
| Vehicle     | variable     |

### Tower
| Description | BYTE_LENGTH |
|-------------|-------------|
| cameras     | variable    |
| lidars      | variable    |
| ...         |             |

### Vehicle
| Description | BYTE_LENGTH  |
|-------------|--------------|
| cameras     | variable     |
| lidars      | variable     |



