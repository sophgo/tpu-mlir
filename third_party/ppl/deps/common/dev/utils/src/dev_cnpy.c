#define _FILE_OFFSET_BITS 64
#define _LARGEFILE64_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <zlib.h>

#define ZIP32_MAX 0xFFFFFFFFULL
#define CRC_CHUNK (1u << 20)

typedef enum {
  CN_DTYPE_FLOAT, CN_DTYPE_INT, CN_DTYPE_UINT, CN_DTYPE_BOOL, CN_DTYPE_COMPLEX
} cn_dtype_class_t;

static int wr16(FILE *fp, uint16_t v){
  unsigned char b[2]={
    v&0xFF,
    (v>>8)&0xFF
  };
  return fwrite(b,1,2,fp)==2?0:-1;
}
static int wr32(FILE *fp, uint32_t v){
  unsigned char b[4]={
    v&0xFF,(v>>8)&0xFF,
    (v>>16)&0xFF,(v>>24)&0xFF
  };
  return fwrite(b,1,4,fp)==4?0:-1; }
static int wr64(FILE *fp, uint64_t v){
  unsigned char b[8]={
    v&0xFF,(v>>8)&0xFF,
    (v>>16)&0xFF,(v>>24)&0xFF,
    (v>>32)&0xFF,(v>>40)&0xFF,
    (v>>48)&0xFF,(v>>56)&0xFF
  };
  return fwrite(b,1,8,fp)==8?0:-1; }
static uint16_t rd16(const unsigned char *p){
  return (uint16_t)p[0]|
         ((uint16_t)p[1]<<8);
}
static uint32_t rd32(const unsigned char *p){
  return (uint32_t)p[0]|
         ((uint32_t)p[1]<<8)|
         ((uint32_t)p[2]<<16)|
         ((uint32_t)p[3]<<24);
}
static uint64_t rd64(const unsigned char *p){
  return (uint64_t)p[0]|
         ((uint64_t)p[1]<<8)|
         ((uint64_t)p[2]<<16)|
         ((uint64_t)p[3]<<24)|
         ((uint64_t)p[4]<<32)|
         ((uint64_t)p[5]<<40)|
         ((uint64_t)p[6]<<48)|
         ((uint64_t)p[7]<<56);
}

int cnpy_build_shape(char *out, size_t out_sz, const size_t *dims, size_t ndims){
  if(!out||out_sz<4||(ndims>0&&!dims))return -1;
  size_t pos=0; out[pos++]='(';
  if(ndims==0){ int n=snprintf(out+pos,out_sz-pos,"1,"); if(n<=0)return -1; pos+=n; }
  else{
    for(size_t i=0;i<ndims;i++){
      int n=snprintf(out+pos,out_sz-pos,"%zu",dims[i]); if(n<=0)return -1; pos+=n;
      if(i+1<ndims){ if(pos+2>=out_sz)return -1; out[pos++]=','; out[pos++]=' '; }
    }
    if(ndims==1){ if(pos+1>=out_sz)return -1; out[pos++]=','; }
  }
  if(pos+2>=out_sz)return -1; out[pos++]=')'; out[pos]='\0'; return 0;
}

static char endian_char(void){ uint16_t x=1; return (*(uint8_t*)&x)==1?'<':'>'; }
static void make_descr(char *out,size_t out_sz,cn_dtype_class_t cls,size_t elem_size){
  char t='f';
  if(cls==CN_DTYPE_INT)t='i'; else if(cls==CN_DTYPE_UINT)t='u';
  else if(cls==CN_DTYPE_BOOL)t='b'; else if(cls==CN_DTYPE_COMPLEX)t='c';
  snprintf(out,out_sz,"%c%c%zu",endian_char(),t,elem_size);
}

static size_t make_npy_header(const char *shape,const char *descr,char **hdr_out){
  if(!shape||!descr||!hdr_out)return 0;
  char dict[2048];
  int n=snprintf(dict,sizeof(dict),"{'descr': '%s', 'fortran_order': False, 'shape': %s, }",descr,shape);
  if(n<=0) return 0;
  size_t dict_len=(size_t)n;
  size_t pad = (16 - (10 + dict_len) % 16) % 16;
  size_t total = dict_len + pad + 1; /* + '\n' */
  if(total>0xFFFF) return 0;

  char *buf=(char*)malloc(10+total);
  if(!buf) return 0;
  size_t pos=0;
  buf[pos++]=(char)0x93; memcpy(buf+pos,"NUMPY",5); pos+=5;
  buf[pos++]=(char)0x01; buf[pos++]=(char)0x00;
  buf[pos++]=(char)(total & 0xFF); buf[pos++]=(char)((total>>8)&0xFF);
  memcpy(buf+pos,dict,dict_len); pos+=dict_len;
  memset(buf+pos,' ',pad); pos+=pad;
  buf[pos++]='\n';
  *hdr_out=buf;
  return pos;
}

static int read_existing_cdir(FILE *fp, char **cdir_buf, size_t *cdir_size, off_t *cdir_off, uint16_t *nrecs){
  *cdir_buf=NULL; *cdir_size=0; *cdir_off=0; *nrecs=0;
  unsigned char eocd[22];
  if(fseeko(fp,-22,SEEK_END)!=0) return -1;
  if(fread(eocd,1,22,fp)!=22) return -1;
  if(!(eocd[0]=='P'&&eocd[1]=='K'&&eocd[2]==0x05&&eocd[3]==0x06)) return -1;
  uint16_t recs = rd16(&eocd[10]);
  uint32_t size32 = rd32(&eocd[12]);
  uint32_t off32  = rd32(&eocd[16]);
  uint16_t comment = rd16(&eocd[20]);
  if(comment!=0) return -1;

  uint64_t size=size32, off=off32;
  if(size32==0xFFFFFFFFu || off32==0xFFFFFFFFu || recs==0xFFFFu){
    unsigned char loc[20];
    if(fseeko(fp,-42,SEEK_END)!=0) return -1;
    if(fread(loc,1,20,fp)!=20) return -1;
    if(!(loc[0]=='P'&&loc[1]=='K'&&loc[2]==0x06&&loc[3]==0x07)) return -1;
    uint64_t zip64_end = rd64(&loc[8]);
    unsigned char z64[56];
    if(fseeko(fp,(off_t)zip64_end,SEEK_SET)!=0) return -1;
    if(fread(z64,1,56,fp)!=56) return -1;
    if(!(z64[0]=='P'&&z64[1]=='K'&&z64[2]==0x06&&z64[3]==0x06)) return -1;
    size = rd64(&z64[40]);
    off  = rd64(&z64[48]);
    recs = (uint16_t)rd64(&z64[32]);
  }

  if(fseeko(fp,(off_t)off,SEEK_SET)!=0) return -1;
  char *buf=(char*)malloc((size_t)size);
  if(!buf) return -1;
  if(fread(buf,1,(size_t)size,fp)!=(size_t)size){ free(buf); return -1; }

  *cdir_buf=buf; *cdir_size=(size_t)size; *cdir_off=(off_t)off; *nrecs=recs;
  if(fseeko(fp,(off_t)off,SEEK_SET)!=0){ free(buf); return -1; }
  return 0;
}

int cnpy_npz_add(const char *zipname,
                 const char *varname,
                 const void *data_ptr,
                 size_t elem_size,
                 size_t elem_count,
                 const char *shape_str,
                 cn_dtype_class_t dtype_class,
                 const char *mode){
  if(!zipname||!varname||!data_ptr||!shape_str||!mode||elem_size==0) return -1;

  char fname[256];
  int flen = snprintf(fname,sizeof(fname),"%s.npy",varname);
  if(flen<=0 || flen>0xFFFF || flen>=(int)sizeof(fname)) return -1;
  uint16_t name_len=(uint16_t)flen;

  /* open file */
  FILE *fp=NULL;
  char *old_cdir=NULL;
  size_t old_cdir_size=0;
  off_t old_cdir_off=0;
  uint16_t old_recs=0;

  if(strcmp(mode,"w")==0){
    fp=fopen(zipname,"wb");
    if(!fp) return -1;
  }else if(strcmp(mode,"a")==0){
    fp=fopen(zipname,"r+b");
    if(!fp){
      fp=fopen(zipname,"wb");
      if(!fp) return -1;
    }else{
      if(read_existing_cdir(fp,&old_cdir,&old_cdir_size,&old_cdir_off,&old_recs)!=0){
        fclose(fp); return -1;
      }
    }
  }else{
    return -1;
  }

  char descr[64]; make_descr(descr,sizeof(descr),dtype_class,elem_size);
  char *npy_hdr=NULL; size_t npy_sz=make_npy_header(shape_str,descr,&npy_hdr);
  if(npy_sz==0 || !npy_hdr){ if(fp)fclose(fp); if(old_cdir)free(old_cdir); return -1; }

  uint64_t data_sz=(uint64_t)elem_size*(uint64_t)elem_count;
  uint64_t total_sz = (uint64_t)npy_sz + data_sz;

  uint32_t crc=crc32(0L,Z_NULL,0);
  crc=crc32(crc,(const Bytef*)npy_hdr,(uInt)npy_sz);
  const unsigned char *p=(const unsigned char*)data_ptr;
  uint64_t rem=data_sz;
  while(rem){
    uInt n=(uInt)((rem>CRC_CHUNK)?CRC_CHUNK:rem);
    crc=crc32(crc,p,n); p+=n; rem-=n;
  }

  off_t local_off=ftello(fp);
  if(local_off<0){ free(npy_hdr); if(old_cdir)free(old_cdir); fclose(fp); return -1; }

  int need_zip64_sizes = (total_sz>ZIP32_MAX);
  uint16_t extra_len = need_zip64_sizes ? (uint16_t)(2+2+8+8) : 0;

  fwrite("PK",1,2,fp);
  wr16(fp,0x0403);
  wr16(fp, need_zip64_sizes?45:20);
  wr16(fp,0); wr16(fp,0); wr16(fp,0); wr16(fp,0);
  wr32(fp,crc);
  wr32(fp, need_zip64_sizes?0xFFFFFFFFu:(uint32_t)total_sz);
  wr32(fp, need_zip64_sizes?0xFFFFFFFFu:(uint32_t)total_sz);
  wr16(fp,name_len);
  wr16(fp,extra_len);
  fwrite(fname,1,name_len,fp);
  if(need_zip64_sizes){
    wr16(fp,0x0001); wr16(fp,16);
    wr64(fp,total_sz); wr64(fp,total_sz);
  }

  fwrite(npy_hdr,1,npy_sz,fp); free(npy_hdr);
  p=(const unsigned char*)data_ptr; rem=data_sz;
  while(rem){
    size_t chunk=(rem>CRC_CHUNK)?CRC_CHUNK:(size_t)rem;
    if(fwrite(p,1,chunk,fp)!=chunk){ if(old_cdir)free(old_cdir); fclose(fp); return -1; }
    p+=chunk; rem-=chunk;
  }

  off_t cdir_start=ftello(fp);
  if(cdir_start<0){ if(old_cdir)free(old_cdir); fclose(fp); return -1; }

  int need_zip64_entry = (total_sz>ZIP32_MAX) || ((uint64_t)local_off>ZIP32_MAX);
  uint16_t zip64_payload=0;
  if(total_sz>ZIP32_MAX) zip64_payload+=16;
  if((uint64_t)local_off>ZIP32_MAX) zip64_payload+=8;
  uint16_t cd_extra_len = need_zip64_entry ? (uint16_t)(2+2+zip64_payload) : 0;

  fwrite("PK",1,2,fp);
  wr16(fp,0x0201);
  wr16(fp, need_zip64_entry?45:20);
  wr16(fp, need_zip64_entry?45:20);
  wr16(fp,0); wr16(fp,0); wr16(fp,0); wr16(fp,0);
  wr32(fp,crc);
  wr32(fp, (total_sz>ZIP32_MAX)?0xFFFFFFFFu:(uint32_t)total_sz);
  wr32(fp, (total_sz>ZIP32_MAX)?0xFFFFFFFFu:(uint32_t)total_sz);
  wr16(fp,name_len);
  wr16(fp,cd_extra_len);
  wr16(fp,0); wr16(fp,0); wr16(fp,0);
  wr32(fp,0);
  wr32(fp, ((uint64_t)local_off>ZIP32_MAX)?0xFFFFFFFFu:(uint32_t)local_off);
  fwrite(fname,1,name_len,fp);
  if(need_zip64_entry){
    wr16(fp,0x0001); wr16(fp,zip64_payload);
    if(total_sz>ZIP32_MAX){ wr64(fp,total_sz); wr64(fp,total_sz); }
    if((uint64_t)local_off>ZIP32_MAX){ wr64(fp,(uint64_t)local_off); }
  }

  if(old_cdir && old_cdir_size>0){
    fwrite(old_cdir,1,old_cdir_size,fp);
    free(old_cdir); old_cdir=NULL;
  }

  uint64_t new_cd_entry_size = 46 + (uint64_t)name_len + (uint64_t)cd_extra_len;
  uint64_t total_cdir_size = new_cd_entry_size + (uint64_t)old_cdir_size;
  uint64_t cdir_offset = (uint64_t)cdir_start;
  uint16_t nrecs_total = (uint16_t)(old_recs + 1);

  int need_zip64_eocd = (cdir_offset>ZIP32_MAX) || (total_cdir_size>ZIP32_MAX) || ((uint64_t)nrecs_total>0xFFFFu);
  if(need_zip64_eocd){
    off_t z64_off=ftello(fp);
    fwrite("PK",1,2,fp); wr16(fp,0x0606);
    wr64(fp,44); wr16(fp,45); wr16(fp,45);
    wr32(fp,0); wr32(fp,0);
    wr64(fp,(uint64_t)nrecs_total); wr64(fp,(uint64_t)nrecs_total);
    wr64(fp,total_cdir_size); wr64(fp,cdir_offset);
    fwrite("PK",1,2,fp); wr16(fp,0x0706);
    wr32(fp,0); wr64(fp,(uint64_t)z64_off); wr32(fp,1);
  }

  fwrite("PK",1,2,fp); wr16(fp,0x0605);
  wr16(fp,0); wr16(fp,0);
  wr16(fp,nrecs_total); wr16(fp,nrecs_total);
  wr32(fp, (uint32_t)((total_cdir_size>ZIP32_MAX)?0xFFFFFFFFu:(uint32_t)total_cdir_size));
  wr32(fp, (uint32_t)((cdir_offset>ZIP32_MAX)?0xFFFFFFFFu:(uint32_t)cdir_offset));
  wr16(fp,0);

  fclose(fp);
  return 0;
}
