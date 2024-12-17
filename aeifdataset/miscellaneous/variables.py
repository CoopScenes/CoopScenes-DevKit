# Named constants
SHA256_CHECKSUM_LENGTH = 32  # Length of SHA-256 checksum
INT_LENGTH = 4  # Length of an integer file length


class Config:
    """
    Configuration class to manage library-wide settings.
    """
    REPACK = False  # True when Image and Points should be passed through
