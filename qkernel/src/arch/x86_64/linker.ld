ENTRY(_start)

SECTIONS {
    . = 0x4000000000 + SIZEOF_HEADERS;

    .text :
    {
        *(.text)
    }

    .data :
    {
        *(.data)
    }

}
