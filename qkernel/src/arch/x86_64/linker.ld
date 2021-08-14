ENTRY(_start)

SECTIONS {
    . = 0x8000000000 + SIZEOF_HEADERS;

    .text :
    {
        *(.text)
    }

    .data :
    {
        *(.data)
    }

}
