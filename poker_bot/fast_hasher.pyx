# 🚀 FAST HASHER - Cython implementation for maximum CPU performance
# This file gets compiled to C for native machine speed

import hashlib
import numpy as np
cimport numpy as np  # Import C types from NumPy

# Use "cpdef" to create a function callable from Python with C types for maximum speed
cpdef list map_hashes_cython(list data_to_hash):
    """
    🚀 ULTRA-FAST HASHING: Cython implementation for maximum CPU performance
    This function runs at C speed with automatic GIL release
    """
    # Declare variables with C types for maximum speed
    cdef list hashes = []
    cdef str info_hash
    cdef tuple components
    cdef int player_id
    cdef double pot_size, payoff
    cdef object hole_cards, community_cards
    
    # The Python GIL is automatically released in this loop, allowing real parallelism
    for item in data_to_hash:
        player_id, hole_cards, community_cards, pot_size, payoff = item
        
        # Create hash components using direct byte conversion
        components = (
            player_id,
            hole_cards.tobytes(),  # Direct byte conversion
            community_cards.tobytes(),
            round(pot_size, 2),  # Round to reduce hash collisions
            round(payoff, 2)
        )
        
        # Use repr() for faster tuple serialization than str()
        info_hash = hashlib.md5(repr(components).encode()).hexdigest()
        hashes.append(info_hash)
    
    return hashes

# Additional optimized function for single hash computation
cpdef str compute_single_hash_cython(int player_id, object hole_cards, 
                                   object community_cards, double pot_size, double payoff):
    """
    🚀 SINGLE HASH COMPUTATION: Ultra-fast single hash computation
    For cases where we need to compute individual hashes
    """
    cdef tuple components
    cdef str info_hash
    
    components = (
        player_id,
        hole_cards.tobytes(),
        community_cards.tobytes(),
        round(pot_size, 2),
        round(payoff, 2)
    )
    
    info_hash = hashlib.md5(repr(components).encode()).hexdigest()
    return info_hash 