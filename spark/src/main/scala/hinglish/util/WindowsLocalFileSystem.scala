package hinglish.util

import org.apache.hadoop.fs.permission.FsPermission
import org.apache.hadoop.fs.{FileStatus, LocalFileSystem, Path, RawLocalFileSystem}

import java.io.{FileNotFoundException, IOException}

/** Local Hadoop FileSystem for running Spark on Windows WITHOUT winutils.exe.
  *
  * Hadoop's RawLocalFileSystem shells out to `winutils.exe chmod/chown` on
  * Windows whenever Spark writes files (e.g. `PipelineModel.save`). Rather
  * than downloading a third-party native binary, permission/ownership changes
  * are made no-ops: they carry no meaning for a single-user local NTFS
  * project directory. Data bytes are written by the normal Java I/O path.
  *
  * Only installed when running on Windows (see SparkSessions.builder); Linux /
  * Docker uses Hadoop's default local filesystem unchanged.
  */
class NoPermissionRawLocalFileSystem extends RawLocalFileSystem {
  override def setPermission(p: Path, permission: FsPermission): Unit = ()
  override def setOwner(p: Path, username: String, groupname: String): Unit = ()
  override def mkdirs(f: Path, permission: FsPermission): Boolean = super.mkdirs(f)

  /** Hadoop's status object lazily shells out to `winutils ls` for
    * permissions/owner when they are first read (e.g. while listing input
    * splits to LOAD a saved model). Return a plain, fully-populated status. */
  override def getFileStatus(f: Path): FileStatus = {
    val file = pathToFile(f)
    if (!file.exists()) throw new FileNotFoundException(s"File $f does not exist")
    val isDir = file.isDirectory
    new FileStatus(
      if (isDir) 0L else file.length(),
      isDir,
      1,
      getDefaultBlockSize(f),
      file.lastModified(),
      0L,
      new FsPermission(if (isDir) "755" else "644"),
      sys.props.getOrElse("user.name", "user"),
      "users",
      f.makeQualified(getUri, getWorkingDirectory)
    )
  }

  /** Hadoop's implementation calls FileUtil.list -> FileUtil.canRead ->
    * NativeIO.Windows.access0 (needs hadoop.dll). Plain java.io listing is
    * equivalent for a local directory. */
  override def listStatus(f: Path): Array[FileStatus] = {
    val local = pathToFile(f)
    if (!local.exists()) throw new FileNotFoundException(s"File $f does not exist")
    if (local.isFile) Array(getFileStatus(f))
    else {
      val names = local.list()
      if (names == null) throw new IOException(s"Cannot list $f")
      names.sorted.map(n => getFileStatus(new Path(f, n)))
    }
  }
}

class WindowsLocalFileSystem extends LocalFileSystem(new NoPermissionRawLocalFileSystem)
