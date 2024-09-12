import sys, os

kMemBlockOffset_HwCryptoKeyBlob = 0x0
kMemBlockOffset_EKIB0           = 0x400
kMemBlockOffset_EPRDB0          = 0x480
kMemBlockOffset_EKIB1           = 0x800
kMemBlockOffset_EPRDB1          = 0x880

kMemBlockOffsetToIvt_DCD  = 0x40
kMemBlockOffsetToIvt_XMCD = 0x40

kMemBlockSize_HwCryptoKeyBlob = 0x100
kMemBlockSize_NFCB            = 0x400
kMemBlockSize_DBBT            = 0x420
kMemBlockSize_MBRDPT          = 0x200
kMemBlockSize_FDCB            = 0x200
kMemBlockSize_EKIB            = 0x20
kMemBlockSize_EPRDB           = 0x100
kMemBlockSize_IVT             = 0x20
kMemBlockSize_BootData        = 0x10
kMemBlockSize_CSF             = 0x1000
kMemBlockSize_HabKeyBlob      = 0x200

kMemBlockSize_Container       = 0x2000
kMemBlockSize_Edgelock        = 0x8000

kBootHeaderTag_IVT     = 0xD1
kBootHeaderTag_DCD     = 0xD2
kBootHeaderTagVer_XMCD = 0xC0

kMemberOffsetInIvt_Hdr      = 0x00
kMemberOffsetInIvt_Tag      = 0x00
kMemberOffsetInIvt_Len      = 0x02
kMemberOffsetInIvt_Entry    = 0x04
kMemberOffsetInIvt_Dcd      = 0x0c
kMemberOffsetInIvt_BootData = 0x10
kMemberOffsetInIvt_Self     = 0x14
kMemberOffsetInIvt_Csf      = 0x18

kMemberOffsetInBootData_Start  = 0x00
kMemberOffsetInBootData_Size   = 0x04
kMemberOffsetInBootData_Plugin = 0x08

kMemberOffsetInDcd_Tag        = 0x00
kMemberOffsetInXmcd_TagVer    = 0x03
kMemberOffsetInXmcd_BlockSize = 0x00

kMemberOffsetInContainer_Len      = 0x01
kMemberOffsetInContainer_Tag      = 0x03
kMemberOffsetInContainer_AppNum   = 0x0b

kMemberOffsetInContainer_Img0Offset    = 0x10
kMemberOffsetInContainer_Img0Size      = 0x14
kMemberOffsetInContainer_Img0LoadAddr  = 0x18
kMemberOffsetInContainer_Img0Entry     = 0x20

kFlexramBankSize = 0x8000
kFlexramBankId_Ocram = 0x1
kFlexramBankId_Dtcm  = 0x2
kFlexramBankId_Itcm  = 0x3

